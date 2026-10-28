"""Portable all-read CPU prediction using a saved context-reference pipeline.

No labels, fitting, calibration or dataset-specific adjustments are performed.
Development bundles require an explicit opt-in and are identified in manifests.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
from pathlib import Path

import numpy as np
import psutil
import torch

from common import Measure, save_json
from context_features import transform_reference
from data import encode_context, iter_sites, sha256, summarise
from model import load_model, read_transform, source_sha256, summary_transform

HEADER = ["transcript_id", "transcript_position", "score"]


@torch.inference_mode()
def score_batch(model, sites, context_reference=None):
    if not sites:
        raise ValueError("Cannot score an empty batch")
    model.eval()
    raw_summary = np.stack([summarise(row[3]) for row in sites])
    sequences = np.stack([encode_context(row[2]) for row in sites])
    summaries = summary_transform(raw_summary)
    extra = None
    if model.context_mode == "relative":
        extra_array = np.asarray(transform_reference(context_reference, sequences, raw_summary), dtype=np.float32)
        if extra_array.shape != (len(sites), 9) or not np.isfinite(extra_array).all():
            raise ValueError("Invalid transformed context-reference features")
        extra = torch.from_numpy(extra_array)
    if model.config["pool"] == "summary":
        reads = np.zeros((len(sites), 1, 9), dtype=np.float32)
        mask = np.ones((len(sites), 1), dtype=bool)
    else:
        maximum = max(len(row[3]) for row in sites)
        reads = np.zeros((len(sites), maximum, 9), dtype=np.float32)
        mask = np.zeros((len(sites), maximum), dtype=bool)
        for index, (_, _, _, bag) in enumerate(sites):
            reads[index, :len(bag)] = read_transform(bag)
            mask[index, :len(bag)] = True
    scores = torch.sigmoid(model(torch.from_numpy(reads), torch.from_numpy(mask),
        torch.from_numpy(sequences), torch.from_numpy(summaries), extra)).cpu().numpy()
    if scores.shape != (len(sites),) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Model returned invalid probabilities")
    return scores


def validate_csv(path, expected_keys):
    seen = set()
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != HEADER:
            raise ValueError("Prediction header differs from the three-column contract")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Malformed prediction CSV")
            key = row["transcript_id"], int(row["transcript_position"])
            probability = float(row["score"])
            if key in seen or not np.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("Duplicate key or invalid CSV score")
            seen.add(key)
    if seen != expected_keys:
        raise ValueError("Saved CSV does not have complete unique input-site coverage")
    return len(seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-padded-reads", type=int, default=65536,
                        help="Flush batches before N_sites x max_reads exceeds this budget; single sites keep all reads")
    parser.add_argument("--allow-development-model", action="store_true")
    args = parser.parse_args()
    if min(args.threads, args.batch_size, args.max_padded_reads) < 1:
        raise ValueError("Threads, batch size and read budget must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    source = args.input.resolve()
    artifact = args.model.resolve()
    output = args.output.resolve()
    sidecar = output.with_suffix(output.suffix + ".manifest.json")
    temporary = output.with_suffix(output.suffix + ".tmp")
    if {source, artifact} & {output, sidecar, temporary}:
        raise ValueError("Prediction output/manifest/temporary path must not overwrite model or input")
    code_files = ("predict.py", "model.py", "data.py", "context_features.py", "common.py")
    code_hashes = {name: source_sha256(Path(__file__).resolve().parent / name) for name in code_files}
    with Measure() as measure:
        model_hash, input_hash = sha256(artifact), sha256(source)
        model, bundle = load_model(artifact, allow_development_model=args.allow_development_model)
        if output.exists() or sidecar.exists():
            if not output.exists() or not sidecar.exists():
                raise ValueError("Incomplete prior prediction; preserve it and use a fresh output filename")
            previous = json.loads(sidecar.read_text(encoding="utf-8"))
            if (previous.get("status") != "complete" or previous.get("input_sha256") != input_hash
                    or previous.get("model_sha256") != model_hash or previous.get("output_sha256") != sha256(output)
                    or previous.get("code_sha256") != code_hashes):
                raise ValueError("Existing output fingerprints differ; refusing overwrite")
            print("PASS: reused complete prediction with unchanged input/model/output/code fingerprints")
            return
        if temporary.exists():
            raise ValueError("Interrupted temporary CSV exists; preserve it and use a fresh output filename")
        output.parent.mkdir(parents=True, exist_ok=True)
        keys = set()
        total_reads = 0
        n_sites = 0
        batches = 0
        max_observed_padded_reads = 0
        with temporary.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(HEADER)
            pending = []

            def write_batch(rows):
                nonlocal n_sites, total_reads, batches, max_observed_padded_reads
                scores = score_batch(model, rows, bundle.get("context_reference"))
                for row, score in zip(rows, scores):
                    key = row[0], row[1]
                    if key in keys:
                        raise ValueError("Duplicate input transcript-position key")
                    keys.add(key)
                    writer.writerow([key[0], key[1], format(float(score), ".17g")])
                    n_sites += 1
                    total_reads += len(row[3])
                batches += 1
                max_observed_padded_reads = max(max_observed_padded_reads,
                                               len(rows) * max(len(row[3]) for row in rows))

            for site in iter_sites(source):
                next_size = len(pending) + 1
                next_max = max([len(site[3])] + [len(row[3]) for row in pending])
                if pending and (next_size > args.batch_size or next_size * next_max > args.max_padded_reads):
                    write_batch(pending)
                    pending = []
                pending.append(site)
            if pending:
                write_batch(pending)
            handle.flush()
            os.fsync(handle.fileno())
        if n_sites == 0 or validate_csv(temporary, keys) != n_sites:
            raise ValueError("Empty or incomplete prediction")
        if sha256(artifact) != model_hash or sha256(source) != input_hash:
            raise ValueError("Model/input changed during prediction")
        if any(source_sha256(Path(__file__).resolve().parent / name) != expected for name, expected in code_hashes.items()):
            raise ValueError("Prediction source changed during execution")
        if output.exists() or sidecar.exists():
            raise ValueError("Destination appeared during prediction; refusing overwrite")
        os.replace(temporary, output)
    manifest = {"status": "complete", "kind": bundle["kind"], "stage": bundle["stage"],
        "context_mode": model.context_mode, "allow_development_model": args.allow_development_model,
        "input_basename": source.name, "model_basename": artifact.name, "output_basename": output.name,
        "input_sha256": input_hash, "model_sha256": model_hash, "output_sha256": sha256(output),
        "code_sha256": code_hashes, "n_sites": n_sites, "n_reads": total_reads,
        "exact_input_key_coverage": True, "all_reads_used": True, "labels_read": False,
        "threads": args.threads, "batch_size_limit": args.batch_size,
        "max_padded_reads_budget": args.max_padded_reads,
        "maximum_observed_padded_reads": max_observed_padded_reads, "n_batches": batches,
        "resources": measure.record,
        "measurement_scope": "input/model hashing and loading, complete parsing, prediction, CSV verification/write; imports excluded",
        "hardware": {"platform": platform.platform(), "processor": platform.processor(),
            "logical_cpus": os.cpu_count(), "total_ram_bytes": psutil.virtual_memory().total},
        "versions": {"torch": str(torch.__version__), "numpy": np.__version__}}
    save_json(sidecar, manifest)
    print(f"PASS: {n_sites:,} sites / {total_reads:,} reads, stage {bundle['stage']}, all-read prediction")


if __name__ == "__main__":
    main()
