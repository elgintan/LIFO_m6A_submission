"""Training-only sequence-relative signal features, requiring only NumPy.

A reference is the per-feature median of *site* medians, not a read-weighted
average or a chemically unmodified control. For a site, prefer a supported
exact seven-base context, then its central five bases, then the global fit
population. The resulting nine features supplement absolute signal inputs.

No labels, transcript IDs, positions or external data enter this module.
"""
from __future__ import annotations

import hashlib
from numbers import Integral

import numpy as np


VERSION = 1
STD_EPSILON = 1e-8
MEDIAN_SLICE = slice(36, 45)
READ_FEATURE_NAMES = (
    "dwell_time_minus1", "signal_std_minus1", "mean_current_minus1",
    "dwell_time_center", "signal_std_center", "mean_current_center",
    "dwell_time_plus1", "signal_std_plus1", "mean_current_plus1",
)


def _arrays(sequence, summaries):
    """Check schema first; callers select fit rows before checking their values."""
    try:
        sequence = np.asarray(sequence)
        summaries = np.asarray(summaries)
    except (TypeError, ValueError) as error:
        raise ValueError("sequence and summaries must be rectangular arrays") from error
    if sequence.ndim != 2 or sequence.shape[1] != 7:
        raise ValueError("sequence must have shape (n_sites, 7)")
    if sequence.dtype.kind not in "iu":
        raise ValueError("sequence must contain integer codes, not booleans, strings or floats")
    if summaries.ndim != 2 or summaries.shape[1] != 63:
        raise ValueError("summaries must have shape (n_sites, 63)")
    if summaries.dtype.kind not in "iuf":
        raise ValueError("summaries must contain real numbers, not booleans or strings")
    if sequence.shape[0] != summaries.shape[0]:
        raise ValueError("sequence and summaries must have the same number of sites")
    return sequence, summaries


def _values(sequence, summaries):
    if np.any(sequence < 0) or np.any(sequence > 4):
        raise ValueError("sequence codes must be in [0, 4]; 4 is the explicit ambiguity code")
    if not np.isfinite(summaries).all():
        raise ValueError("summaries must be finite")
    return sequence, summaries


def _indices(fit_indices, n_sites):
    try:
        indices = np.asarray(fit_indices)
    except (TypeError, ValueError) as error:
        raise ValueError("fit_indices must be a one-dimensional integer array") from error
    if indices.ndim != 1 or len(indices) == 0 or indices.dtype.kind not in "iu":
        raise ValueError("fit_indices must be a non-empty one-dimensional integer array")
    if np.any(indices < 0) or np.any(indices >= n_sites):
        raise ValueError("fit_indices contains an out-of-range site index")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("fit_indices must not contain duplicate sites")
    # A partition's statistics/hash must not depend on the supplied row order.
    return np.sort(indices.astype(np.int64, copy=False))


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _context_keys(sequence):
    return ["".join(str(int(code)) for code in row) for row in sequence]


def _fit_groups(keys, site_medians, min_sites):
    groups = {}
    for row, key in enumerate(keys):
        groups.setdefault(key, []).append(row)
    counts = {key: len(groups[key]) for key in sorted(groups)}
    references = {
        key: np.median(site_medians[groups[key]], axis=0).tolist()
        for key in sorted(groups) if counts[key] >= min_sites
    }
    return references, counts


def _raw_residuals(reference, sequence, site_medians):
    baselines = np.empty_like(site_medians, dtype=np.float64)
    exact = reference["exact7_refs"]
    central = reference["central5_refs"]
    global_reference = reference["global_reference"]
    for row, key in enumerate(_context_keys(sequence)):
        baselines[row] = exact.get(key, central.get(key[1:6], global_reference))
    with np.errstate(over="raise", invalid="raise"):
        try:
            residuals = site_medians - baselines
        except FloatingPointError as error:
            raise ValueError("raw context residuals overflowed float64") from error
    if not np.isfinite(residuals).all():
        raise ValueError("raw context residuals must be finite")
    return residuals


def _signed_signal_transform(residuals):
    """Match the old read-transform formulas, operating on raw residuals."""
    transformed = np.array(residuals, dtype=np.float64, copy=True)
    with np.errstate(over="raise", invalid="raise"):
        try:
            for column in (0, 3, 6):
                values = transformed[:, column]
                transformed[:, column] = np.sign(values) * np.log1p(np.abs(values) * 1000.0)
            for column in (1, 4, 7):
                values = transformed[:, column]
                transformed[:, column] = np.sign(values) * np.log1p(np.abs(values))
        except FloatingPointError as error:
            raise ValueError("context residual transform overflowed float64") from error
    if not np.isfinite(transformed).all():
        raise ValueError("transformed context residuals must be finite")
    return transformed


def _vector(values, name, positive=False):
    array = np.asarray(values)
    if array.shape != (9,) or array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"reference {name} must contain nine finite real numbers")
    array = array.astype(np.float64)
    if positive and np.any(array <= 0):
        raise ValueError(f"reference {name} must contain positive scales")
    return array


def _validate_reference(reference):
    if not isinstance(reference, dict) or reference.get("version") != VERSION \
            or type(reference.get("version")) is not int:
        raise ValueError("unsupported context reference version")
    n_fit = _positive_integer(reference.get("n_fit"), "reference n_fit")
    minimum = _positive_integer(reference.get("min_sites"), "reference min_sites")
    _vector(reference.get("global_reference"), "global_reference")
    _vector(reference.get("residual_mean"), "residual_mean")
    _vector(reference.get("residual_sd"), "residual_sd", positive=True)
    for prefix, length in (("exact7", 7), ("central5", 5)):
        counts = reference.get(prefix + "_counts")
        refs = reference.get(prefix + "_refs")
        if not isinstance(counts, dict) or not isinstance(refs, dict) or not counts:
            raise ValueError(f"reference {prefix} counts/refs must be dictionaries")
        for key, count in counts.items():
            if not isinstance(key, str) or len(key) != length or any(c not in "01234" for c in key):
                raise ValueError(f"reference {prefix} contains an invalid encoded context key")
            _positive_integer(count, f"reference {prefix} count")
        if sum(counts.values()) != n_fit:
            raise ValueError(f"reference {prefix} counts do not total n_fit")
        supported = {key for key, count in counts.items() if count >= minimum}
        if set(refs) != supported:
            raise ValueError(f"reference {prefix} refs do not match supported training contexts")
        for key, values in refs.items():
            _vector(values, f"{prefix} reference {key}")
    collapsed = {}
    for key, count in reference["exact7_counts"].items():
        central = key[1:6]
        collapsed[central] = collapsed.get(central, 0) + count
    if collapsed != reference["central5_counts"]:
        raise ValueError("reference central5 counts disagree with exact7 counts")
    fingerprint = reference.get("fit_indices_sha256")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64 \
            or any(c not in "0123456789abcdef" for c in fingerprint):
        raise ValueError("reference fit_indices_sha256 is invalid")


def fit_reference(sequence, summaries, fit_indices, min_sites=30):
    """Fit one label-neutral reference using exactly the permitted site rows.

    Parameters
    ----------
    sequence : integer array (n_sites, 7)
        A/C/G/T-or-U/ambiguity encoded as 0/1/2/3/4.
    summaries : real array (n_sites, 63)
        Raw cached means(9), population SD(9), then nine each of the
        .10/.25/.50/.75/.90 quantiles. Site medians are columns [36:45].
    fit_indices : one-dimensional, unique, non-empty integer array
        Permitted training sites only. Other rows do not enter references,
        counts, centre/scale fitting, or per-row value validation.
    min_sites : positive integer, default 30
        Minimum *sites*, not reads, for either context-specific reference.

    Returns
    -------
    dict
        JSON-serializable context references, counts, residual standardisation
        and complete feature definitions. Refit independently in every
        permitted training partition; save the final instance with its model.
    """
    sequence, summaries = _arrays(sequence, summaries)
    indices = _indices(fit_indices, len(sequence))
    minimum = _positive_integer(min_sites, "min_sites")
    fit_sequence, fit_summaries = _values(sequence[indices], summaries[indices])
    site_medians = fit_summaries[:, MEDIAN_SLICE].astype(np.float64)
    keys7 = _context_keys(fit_sequence)
    try:
        with np.errstate(over="raise", invalid="raise"):
            exact_refs, exact_counts = _fit_groups(keys7, site_medians, minimum)
            central_refs, central_counts = _fit_groups(
                [key[1:6] for key in keys7], site_medians, minimum)
            global_reference = np.median(site_medians, axis=0)
    except FloatingPointError as error:
        raise ValueError("context reference median calculation overflowed float64") from error
    reference = {
        "version": VERSION,
        "n_fit": len(indices),
        "min_sites": minimum,
        "exact7_refs": exact_refs,
        "exact7_counts": exact_counts,
        "central5_refs": central_refs,
        "central5_counts": central_counts,
        "global_reference": global_reference.tolist(),
        "fit_indices_sha256": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
        "feature_definitions": {
            "raw_summary_shape": ["n_sites", 63],
            "site_median_columns_zero_based": list(range(36, 45)),
            "feature_names_in_order": ["context_residual_" + name for name in READ_FEATURE_NAMES],
            "context_codes": {"A": 0, "C": 1, "G": 2, "T_or_U": 3, "ambiguity": 4},
            "context_key_encoding": "seven/five digit strings, one code 0..4 per base",
            "central5_sequence_columns_zero_based": [1, 2, 3, 4, 5],
            "reference_statistic": "per-feature median of raw per-site medians; equal weight per fit site",
            "fallback_order": ["exact7 with >= min_sites fit sites", "central5 with >= min_sites fit sites", "global fit-site median"],
            "raw_residual": "raw_site_median - selected_raw_context_reference",
            "transforms_by_column": [
                "sign(x)*log1p(abs(x)*1000)", "sign(x)*log1p(abs(x))", "x",
                "sign(x)*log1p(abs(x)*1000)", "sign(x)*log1p(abs(x))", "x",
                "sign(x)*log1p(abs(x)*1000)", "sign(x)*log1p(abs(x))", "x",
            ],
            "normalisation": "(transformed_residual - fit_residual_mean) / fit_population_sd",
            "sd_ddof": 0,
            "near_zero_sd_threshold": STD_EPSILON,
            "near_zero_sd_replacement": 1.0,
            "fit_indices_hash_encoding": "SHA256 of sorted unique signed little-endian int64 row indices",
            "fit_uses_labels_or_ids": False,
            "reference_is_chemically_unmodified_ground_truth": False,
            "calculation_dtype": "float64",
            "output_dtype": "float32",
        },
    }
    transformed = _signed_signal_transform(_raw_residuals(reference, fit_sequence, site_medians))
    try:
        with np.errstate(over="raise", invalid="raise"):
            centre = transformed.mean(axis=0)
            scale = transformed.std(axis=0, ddof=0)
    except FloatingPointError as error:
        raise ValueError("context residual normalisation overflowed float64") from error
    reference["residual_mean"] = centre.tolist()
    reference["residual_sd"] = np.where(scale < STD_EPSILON, 1.0, scale).tolist()
    _validate_reference(reference)
    return reference


def transform_reference(reference, sequence, summaries):
    """Return nine finite float32 features using only the saved fitted bundle.

    All rows supplied here are validated. Sparse or unseen contexts follow the
    saved exact7 -> central5 -> global hierarchy; code 4 remains a valid base.
    No reference is fitted or updated when processing new/held-out sites.
    """
    _validate_reference(reference)
    sequence, summaries = _values(*_arrays(sequence, summaries))
    site_medians = summaries[:, MEDIAN_SLICE].astype(np.float64)
    transformed = _signed_signal_transform(_raw_residuals(reference, sequence, site_medians))
    centre = _vector(reference["residual_mean"], "residual_mean")
    scale = _vector(reference["residual_sd"], "residual_sd", positive=True)
    with np.errstate(over="ignore", invalid="ignore"):
        result = ((transformed - centre) / scale).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("normalised context features cannot be represented as finite float32")
    return result
