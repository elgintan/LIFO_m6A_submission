"""Strict, label-free JSON input and reusable all-read caches.

One input line is one site. Reads remain an unordered N x 9 matrix, with
the -1/centre/+1 event measurements retained in their supplied order.
No fitting, label inspection, filtering, read subsampling or imputation occurs
here. A completed manifest is committed only after a successful EOF scan.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import numpy as np

READ_NAMES = [
    "dwell_time_minus1", "signal_std_minus1", "mean_signal_minus1",
    "dwell_time_center", "signal_std_center", "mean_signal_center",
    "dwell_time_plus1", "signal_std_plus1", "mean_signal_plus1",
]
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
SUMMARY_NAMES = (
    [f"mean_{name}" for name in READ_NAMES]
    + [f"std_{name}" for name in READ_NAMES]
    + [f"q{int(q*100):02d}_{name}" for q in QUANTILES for name in READ_NAMES]
)
BASE_CODES = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}
UNKNOWN_IUPAC = frozenset("RYSWKMBDHVN")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"non-finite JSON numeric constant {value}")


def encode_context(context: str) -> np.ndarray:
    """ACGT/U -> 0,1,2,3; ten recognised ambiguous IUPAC letters -> 4."""
    if not isinstance(context, str) or len(context) != 7 or not context.isascii():
        raise ValueError("sequence context must contain exactly seven ASCII IUPAC letters")
    result = []
    for letter in context.upper():
        if letter in BASE_CODES:
            result.append(BASE_CODES[letter])
        elif letter in UNKNOWN_IUPAC:
            result.append(4)
        else:
            raise ValueError(f"unsupported sequence symbol {letter!r}; expected DNA/RNA IUPAC")
    return np.asarray(result, dtype=np.int64)


def _matrix(reads) -> np.ndarray:
    if isinstance(reads, np.ndarray):
        if reads.ndim != 2 or reads.shape[0] == 0 or reads.shape[1] != 9:
            raise ValueError("site must contain a non-empty N x 9 read matrix")
        if reads.dtype.kind not in "iuf":
            raise ValueError("read measurements must be real numbers, never booleans or strings")
    else:
        if not isinstance(reads, list) or not reads:
            raise ValueError("site must contain a non-empty list of reads")
        for index, row in enumerate(reads, 1):
            if not isinstance(row, list) or len(row) != 9:
                raise ValueError(f"read {index} must contain exactly nine measurements")
            if any(type(value) not in (int, float) for value in row):
                raise ValueError(f"read {index} contains a boolean, string or non-number")
    try:
        matrix = np.asarray(reads, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("measurements cannot be represented as finite float64 values") from error
    if not np.isfinite(matrix).all():
        raise ValueError("measurements contain NaN or infinity")
    return matrix


def summarise(reads) -> np.ndarray:
    """All-read means, population SD (ddof=0), then five nine-column quantiles.

    Calculation is float64. Returned features are float32 after checking that
    the cast cannot create infinity. NumPy's default linear quantile method is
    specified explicitly for reproducibility.
    """
    matrix = _matrix(reads)
    features64 = np.concatenate((
        matrix.mean(axis=0), matrix.std(axis=0, ddof=0),
        np.quantile(matrix, QUANTILES, axis=0, method="linear").reshape(-1),
    ))
    with np.errstate(over="ignore", invalid="ignore"):
        features = features64.astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("summary cannot be represented as finite float32 features")
    return features


def iter_sites(path: str | Path):
    """Yield (transcript str, position int, context str, all reads float64).

    Duplicate site keys are rejected across the entire stream, including
    numerically equivalent position text such as '01' and '1'. Malformed rows
    stop processing; an empty line is malformed rather than silently skipped.
    """
    path = Path(path)
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    seen = set()
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            description = "unknown site"
            try:
                record = json.loads(line, object_pairs_hook=_unique_object,
                                    parse_constant=_reject_constant)
                if not isinstance(record, dict) or len(record) != 1:
                    raise ValueError("expected exactly one transcript per JSON line")
                transcript, positions = next(iter(record.items()))
                if not transcript or transcript != transcript.strip():
                    raise ValueError("transcript ID is empty or whitespace-padded")
                if not isinstance(positions, dict) or len(positions) != 1:
                    raise ValueError("expected exactly one transcript position per JSON line")
                position_text, contexts = next(iter(positions.items()))
                description = f"{transcript}:{position_text}"
                if not position_text or not position_text.isascii() or not position_text.isdecimal():
                    raise ValueError("transcript position must be a non-negative integer")
                position = int(position_text)
                if position > np.iinfo(np.int64).max:
                    raise ValueError("transcript position exceeds int64 range")
                if not isinstance(contexts, dict) or len(contexts) != 1:
                    raise ValueError("expected exactly one context for the site")
                context, reads = next(iter(contexts.items()))
                encode_context(context)
                matrix = _matrix(reads)
                # Every value retained in the binary cache must remain finite.
                with np.errstate(over="ignore", invalid="ignore"):
                    matrix32 = matrix.astype(np.float32)
                if not np.isfinite(matrix32).all():
                    raise ValueError("read cannot be represented as finite float32 values")
                site_key = (transcript, position)
                if site_key in seen:
                    raise ValueError("duplicate transcript-position site key")
                seen.add(site_key)
            except (ValueError, TypeError, OverflowError) as error:
                raise ValueError(f"JSON line {line_number}, {description}: {error}") from error
            yield transcript, position, context, matrix


def schema() -> dict:
    return {
        "schema_version": 1,
        "read_columns": READ_NAMES,
        "read_storage": "float32 little-endian N x 9; all supplied reads",
        "summary_columns": SUMMARY_NAMES,
        "summary_calculation_dtype": "float64",
        "summary_storage_dtype": "float32",
        "summary_std_ddof": 0,
        "summary_quantiles": list(QUANTILES),
        "summary_quantile_method": "linear",
        "sequence_length": 7,
        "sequence_codes": {"A": 0, "C": 1, "G": 2, "T/U": 3,
                           "R/Y/S/W/K/M/B/D/H/V/N": 4},
        "context_case_policy": "case-insensitive encoding; original text retained in keys.csv",
        "non_numeric_or_empty_or_malformed_policy": "fail; no imputation, dropping or coercion",
        "excluded_predictors": ["transcript_id", "transcript_position", "gene_id",
                                "label", "fold_id", "file_offsets"],
        "learned_preprocessing": "none",
    }


def _atomic_json(path: Path, content: dict):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class _Measurements:
    """Sample this Python process, excluding concurrent installation/training."""
    def __init__(self):
        import psutil
        self.process = psutil.Process()
        self.started = time.perf_counter()
        self.cpu_started = self.process.cpu_times()
        self.peak_rss = self.process.memory_info().rss
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def _sample(self):
        while not self.stop.wait(0.2):
            try:
                self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
            except Exception:
                return

    def finish(self) -> dict:
        self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
        self.stop.set()
        self.thread.join()
        cpu = self.process.cpu_times()
        return {
            "wall_seconds": time.perf_counter() - self.started,
            "process_cpu_seconds": (cpu.user + cpu.system - self.cpu_started.user
                                    - self.cpu_started.system),
            "sampled_peak_process_rss_bytes": self.peak_rss,
            "rss_sampling_interval_seconds": 0.2,
            "rss_scope": "preparation Python process only; sampled, not OS exact maximum",
        }


def prepare(input_path: str | Path, output_dir: str | Path) -> dict:
    """Build label-free cache, or verify and reuse its exact completed cache.

    Interrupted preparation safely restarts from the input; completed caches
    avoid re-parsing. Training checkpoints are handled by the trainer.
    """
    input_path, output_dir = Path(input_path).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    measured = _Measurements()
    input_hash = sha256(input_path)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") == "completed" and existing.get("input_sha256") == input_hash:
            for name, expected in existing["cache_sha256"].items():
                if not (output_dir / name).exists() or sha256(output_dir / name) != expected:
                    measured.finish()
                    raise ValueError(f"completed cache is missing/corrupted: {name}")
            measured.finish()
            print(f"Verified and reused completed cache {output_dir}", flush=True)
            return json.loads((output_dir / "audit.json").read_text(encoding="utf-8"))

    _atomic_json(manifest_path, {"status": "preparing", "input_sha256": input_hash,
                               "input_path": str(input_path), "schema_version": 1})
    names = ("reads.bin", "keys.csv", "offsets.npy", "summary.npy", "seq.npy")
    temporary = {name: output_dir / (name + ".partial") for name in names}
    summary_raw = output_dir / "summary.raw.partial"
    seq_raw = output_dir / "seq.raw.partial"
    counts = []
    offsets = [0]
    unknown_contexts = 0
    transcripts = set()
    contexts_seen = set()
    try:
        with temporary["reads.bin"].open("wb") as read_file, \
                temporary["keys.csv"].open("w", newline="", encoding="utf-8") as key_file, \
                summary_raw.open("wb") as summary_file, seq_raw.open("wb") as seq_file:
            writer = csv.writer(key_file)
            writer.writerow(["transcript_id", "transcript_position", "context", "n_reads"])
            for count, (transcript, position, context, reads) in enumerate(iter_sites(input_path), 1):
                reads.astype("<f4").tofile(read_file)
                summarise(reads).astype("<f4", copy=False).tofile(summary_file)
                seq = encode_context(context)
                seq.astype("<i8", copy=False).tofile(seq_file)
                writer.writerow([transcript, position, context, len(reads)])
                counts.append(len(reads))
                offsets.append(offsets[-1] + len(reads))
                transcripts.add(transcript)
                contexts_seen.add(context.upper().replace("U", "T"))
                unknown_contexts += int(np.any(seq == 4))
                if count % 10000 == 0:
                    print(f"Prepared {count:,} sites / {offsets[-1]:,} reads; "
                          f"elapsed {time.perf_counter() - measured.started:.1f}s", flush=True)
                    key_file.flush()
            if not counts:
                raise ValueError("input contains no sites")
            for handle in (read_file, key_file, summary_file, seq_file):
                handle.flush()
                os.fsync(handle.fileno())

        n_sites, n_reads = len(counts), offsets[-1]
        offsets_array = np.asarray(offsets, dtype=np.int64)
        if len(offsets_array) != n_sites + 1 or offsets_array[0] != 0 \
                or offsets_array[-1] != n_reads or np.any(np.diff(offsets_array) <= 0):
            raise ValueError("invalid offsets: each site must map to all its non-empty reads")
        if temporary["reads.bin"].stat().st_size != n_reads * 9 * 4:
            raise ValueError("read binary byte count does not match recorded shape")
        with temporary["offsets.npy"].open("wb") as handle:
            np.save(handle, offsets_array, allow_pickle=False)
        for raw_path, name, dtype, shape in (
                (summary_raw, "summary.npy", "<f4", (n_sites, 63)),
                (seq_raw, "seq.npy", "<i8", (n_sites, 7))):
            if raw_path.stat().st_size != np.dtype(dtype).itemsize * int(np.prod(shape)):
                raise ValueError(f"invalid cache shape for {name}")
            matrix = np.memmap(raw_path, dtype=dtype, mode="r", shape=shape)
            with temporary[name].open("wb") as handle:
                np.save(handle, matrix, allow_pickle=False)
            del matrix
        # The manifest remains 'preparing' while output files are promoted.
        for name in names:
            os.replace(temporary[name], output_dir / name)
        summary_raw.unlink()
        seq_raw.unlink()
        cache_hashes = {name: sha256(output_dir / name) for name in names}
        count_array = np.asarray(counts, dtype=np.int64)
        measurements = measured.finish()
        audit = {
            "status": "completed", "input_path": str(input_path),
            "input_sha256": input_hash, "input_bytes": input_path.stat().st_size,
            "output_dir": str(output_dir), "n_sites": n_sites,
            "n_reads_total": n_reads, "n_transcripts": len(transcripts),
            "n_distinct_contexts": len(contexts_seen),
            "n_contexts_with_ambiguous_iupac": unknown_contexts,
            "read_count": {"min": int(count_array.min()), "max": int(count_array.max()),
                           "mean": float(count_array.mean()),
                           "quantiles": dict(zip(["p0", "p10", "p25", "p50", "p75", "p90", "p100"],
                                                  np.quantile(count_array, [0, .1, .25, .5, .75, .9, 1]).tolist()))},
            "schema": schema(), "measurements": measurements,
            "cache_sha256": cache_hashes,
            "cache_bytes": {name: (output_dir / name).stat().st_size for name in names},
            "labels_or_fold_ids_read": False,
            "read_sampling": "none; every supplied read cached and used for summaries",
            "malformed_sites_dropped": 0,
            "validation": "strict JSON/numeric/context/unique keys, finite float64 and float32, byte counts and offsets",
        }
        _atomic_json(output_dir / "audit.json", audit)
        _atomic_json(manifest_path, {
            "status": "completed", "input_sha256": input_hash,
            "input_path": str(input_path), "schema_version": 1,
            "n_sites": n_sites, "n_reads_total": n_reads, "cache_sha256": cache_hashes,
        })
        print(f"Completed cache: {n_sites:,} sites, {n_reads:,} reads, "
              f"{measurements['wall_seconds']:.1f}s, "
              f"peak sampled RSS {measurements['sampled_peak_process_rss_bytes']/1024**2:.1f} MiB", flush=True)
        return audit
    except BaseException as error:
        measurements = measured.finish()
        _atomic_json(manifest_path, {
            "status": "failed", "input_sha256": input_hash,
            "input_path": str(input_path), "error": str(error),
            "sites_written": len(counts), "reads_written": offsets[-1],
            "measurements": measurements,
        })
        raise
