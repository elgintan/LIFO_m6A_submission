"""Shared binary scorer: data0 gene-fold OOF development, or exposed data1 transfer.

This checks prediction contents and fold assignments, not whether a model really
excluded assessment genes during training. Retain training code and logs as well.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


KEY_COLUMNS = ["transcript_id", "transcript_position"]
FOLD_SHA256 = "986b7872084cf762f1c3944b1eacc1e8f1ff69cb2848630f80685063541da9b0"
DATA0_LABELS_CANONICAL_SHA256 = "153f4658b82e4f6d95c449723d238b63153cd0102c826a3c61e3a652749b1a87"
DEFAULT_FOLDS = Path(__file__).resolve().parents[1] / "eda/received/shared_folds/data0_folds.csv"
RANKING_METRICS = ("roc_auc", "pr_auc_trapezoid", "average_precision", "selection_score")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def integer(value, name):
    """Accept integral numeric CSV values without rounding fractional positions."""
    try:
        number = Decimal(value)
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not number.is_finite() or number != number.to_integral_value() or number < 0:
        raise ValueError(f"{name} must be a finite nonnegative integer")
    return int(number)


def csv_rows(path, required, exact=False):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if not fields or len(fields) != len(set(fields)):
            raise ValueError(f"{Path(path).name}: missing or duplicate column names")
        if (exact and fields != required) or (not exact and not set(required).issubset(fields)):
            raise ValueError(f"{Path(path).name}: expected columns {required}"
                             + (" in that order, with no extras" if exact else " (extra columns allowed)"))
        for line, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{Path(path).name}:{line}: inconsistent CSV field count")
            yield row


def site_key(row):
    transcript = row["transcript_id"]
    if not transcript or transcript.strip() != transcript:
        raise ValueError("transcript_id must be nonempty and have no surrounding whitespace")
    return transcript, integer(row["transcript_position"], "transcript_position")


def read_labels(path, label_column="label"):
    labels, genes = {}, {}
    for row in csv_rows(path, KEY_COLUMNS + [label_column]):
        key = site_key(row)
        if key in labels:
            raise ValueError("Duplicate site key in labels")
        try:
            label = Decimal(row[label_column])
        except InvalidOperation as exc:
            raise ValueError("Labels must be binary 0 or 1") from exc
        if not label.is_finite() or label not in (0, 1):
            raise ValueError("Labels must be binary 0 or 1; fractional labels are not supported")
        labels[key] = int(label)
        if "gene_id" in row:
            if not row["gene_id"] or row["gene_id"].strip() != row["gene_id"]:
                raise ValueError("gene_id must be nonempty and have no surrounding whitespace")
            genes[key] = row["gene_id"]
    if not labels:
        raise ValueError("Empty label file")
    return labels, genes


def read_predictions(path, dataset):
    columns = KEY_COLUMNS + ["score"] + (["fold_id"] if dataset == "data0" else [])
    predictions, fold_ids = {}, {}
    for row in csv_rows(path, columns, exact=True):
        key = site_key(row)
        if key in predictions:
            raise ValueError("Duplicate site key in predictions")
        try:
            score = float(row["score"])
        except ValueError as exc:
            raise ValueError("Scores must be finite probabilities between 0 and 1") from exc
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Scores must be finite probabilities between 0 and 1")
        predictions[key] = score
        if dataset == "data0":
            fold_ids[key] = integer(row["fold_id"], "fold_id")
    if not predictions:
        raise ValueError("Empty prediction file")
    return predictions, fold_ids


def read_folds(path):
    folds, gene_folds = {}, {}
    for row in csv_rows(path, KEY_COLUMNS + ["gene_id", "fold_id"], exact=True):
        key = site_key(row)
        if key in folds:
            raise ValueError("Duplicate site key in fold manifest")
        gene = row["gene_id"]
        fold = integer(row["fold_id"], "fold_id")
        if not gene or gene.strip() != gene or fold not in range(5):
            raise ValueError("Fold manifest requires nonempty gene_id and fold_id in 0..4")
        if gene in gene_folds and gene_folds[gene] != fold:
            raise ValueError("A gene occurs in more than one fold")
        gene_folds[gene] = fold
        folds[key] = (gene, fold)
    if {fold for _, fold in folds.values()} != set(range(5)):
        raise ValueError("Fold manifest must contain exactly five folds 0..4")
    return folds


def same_keys(reference, observed, description):
    missing = reference.keys() - observed.keys()
    extra = observed.keys() - reference.keys()
    if missing or extra:
        raise ValueError(f"{description}: site coverage differs ({len(missing)} missing, {len(extra)} extra)")


def canonical_labels_hash(labels, folds):
    digest = hashlib.sha256()
    for transcript, position in sorted(labels):
        record = [transcript, position, folds[(transcript, position)][0], labels[(transcript, position)]]
        digest.update((json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8"))
    return digest.hexdigest()


def metrics(y, scores):
    y, scores = np.asarray(y), np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or scores.ndim != 1 or len(y) != len(scores):
        raise ValueError("Labels and scores must be matching one-dimensional arrays")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Binary metrics require both classes in every scored partition")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Scores must be finite probabilities between 0 and 1")
    precision_curve, recall_curve, _ = precision_recall_curve(y, scores, drop_intermediate=False)
    roc = float(roc_auc_score(y, scores))
    pr = float(auc(recall_curve, precision_curve))
    tn, fp, fn, tp = map(int, confusion_matrix(y, scores >= .5, labels=[0, 1]).ravel())
    return {
        "n_sites": len(y), "n_positives": int(y.sum()), "positive_fraction": float(y.mean()),
        "roc_auc": roc, "pr_auc_trapezoid": pr,
        "average_precision": float(average_precision_score(y, scores)),
        "selection_score": (roc + pr) / 2, "threshold": .5,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "precision": tp / (tp + fp) if tp + fp else 0.,
        "recall": tp / (tp + fn), "f1": 2 * tp / (2 * tp + fp + fn),
        "specificity": tn / (tn + fp), "accuracy": (tn + tp) / len(y),
    }


def evaluate_tables(labels, predictions, dataset, folds=None, prediction_folds=None, label_genes=None):
    """Evaluate already parsed tables; production file/hash checks are in evaluate_files."""
    if dataset not in ("data0", "data1"):
        raise ValueError("Only binary data0 and data1 are supported")
    same_keys(labels, predictions, "Labels versus predictions")
    ordered = sorted(labels)
    y = np.asarray([labels[key] for key in ordered])
    scores = np.asarray([predictions[key] for key in ordered])
    result = {
        "dataset": dataset,
        "evaluation_kind": "data0_gene_grouped_oof_development" if dataset == "data0" else "data1_exposed_transfer",
        "pooled": metrics(y, scores),
    }
    if dataset == "data0":
        if folds is None or prediction_folds is None:
            raise ValueError("data0 requires fold manifest and prediction fold_id")
        same_keys(labels, folds, "Labels versus fold manifest")
        same_keys(labels, prediction_folds, "Labels versus prediction fold IDs")
        if any(prediction_folds[key] != folds[key][1] for key in ordered):
            raise ValueError("Prediction fold_id differs from the shared fold manifest")
        if label_genes and any(label_genes[key] != folds[key][0] for key in ordered):
            raise ValueError("Label gene_id differs from the shared fold manifest")
        assigned = np.asarray([folds[key][1] for key in ordered])
        per_fold = [{"fold_id": fold, **metrics(y[assigned == fold], scores[assigned == fold])}
                    for fold in range(5)]
        means = {name: float(np.mean([row[name] for row in per_fold])) for name in RANKING_METRICS}
        deviations = {name: float(np.std([row[name] for row in per_fold], ddof=1)) for name in RANKING_METRICS}
        result.update(per_fold=per_fold, mean_fold=means, sample_sd_fold_ddof1=deviations,
                      primary_score=means["selection_score"],
                      primary_score_definition="Unweighted mean of five fold (ROC AUC + trapezoidal PR AUC) / 2")
    else:
        result.update(primary_score=result["pooled"]["selection_score"],
                      primary_score_definition="Pooled (ROC AUC + trapezoidal PR AUC) / 2; already-exposed transfer diagnostic")
    return result


def evaluate_files(labels_path, predictions_path, dataset, label_column="label", folds_path=DEFAULT_FOLDS):
    inputs = {"labels": Path(labels_path), "predictions": Path(predictions_path)}
    if dataset == "data0":
        inputs["folds"] = Path(folds_path)
    hashes = {name: sha256(path) for name, path in inputs.items()}
    if dataset == "data0" and hashes["folds"] != FOLD_SHA256:
        raise ValueError("Fold manifest SHA256 differs from the agreed immutable file; restore it from GitHub")
    labels, genes = read_labels(labels_path, label_column)
    predictions, prediction_folds = read_predictions(predictions_path, dataset)
    folds = read_folds(folds_path) if dataset == "data0" else None
    result = evaluate_tables(labels, predictions, dataset, folds, prediction_folds, genes)
    if dataset == "data0":
        canonical_hash = canonical_labels_hash(labels, folds)
        if canonical_hash != DATA0_LABELS_CANONICAL_SHA256:
            raise ValueError("data0 IDs/positions/genes/labels differ from the agreed source")
        result["data0_labels_canonical_sha256"] = canonical_hash
    if any(sha256(path) != hashes[name] for name, path in inputs.items()):
        raise ValueError("An input file changed during scoring; no output written")
    result.update(
        schema_version="1.0",
        scorer_sha256=sha256(Path(__file__)),
        created_utc=datetime.now(timezone.utc).isoformat(),
        inputs={name: {"filename": path.name, "sha256": hashes[name]} for name, path in inputs.items()},
        label_column=label_column,
        versions={"python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__},
        limitations=["File validation cannot prove training excluded assessment genes; retain fit logs and code.",
                     "data0 CV used for model selection is development evidence, not an untouched test.",
                     "data1 was previously examined and is not an untouched final test."],
    )
    return result


def write_json(path, result):
    """Replace output atomically, only after all validation and scoring succeed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True, help="Local course label CSV; never upload raw datasets")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset", choices=("data0", "data1"), required=True)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--output", type=Path, required=True, help="JSON metrics output")
    args = parser.parse_args(argv)
    try:
        inputs = [args.labels, args.predictions] + ([args.folds] if args.dataset == "data0" else [])
        if any(args.output.resolve() == path.resolve() for path in inputs):
            raise ValueError("Output must not replace an input file")
        result = evaluate_files(args.labels, args.predictions, args.dataset, args.label_column, args.folds)
        write_json(args.output, result)
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(2, f"Evaluation failed: {exc}\n")
    print(f"{args.dataset}: primary score {result['primary_score']:.10f}; metrics saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
