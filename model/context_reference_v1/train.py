"""Portable data0-only neural fitting with the immutable shared gene folds.

smoke checks the pipeline on a deterministic collection of whole genes;
cv evaluates the complete supplied data0 population; final requires complete CV.
No data1 labels, historical model weights, or assessment-fitted statistics enter
fitting. Checkpoints resume at the last completed epoch, with signatures checked.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import pandas as pd
import sklearn
import torch

from common import Measure, save_json
from context_features import fit_reference, transform_reference
from data import sha256
from model import SiteModel, read_transform, summary_transform, source_sha256

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
SPEC = importlib.util.spec_from_file_location("team_scorer", REPO / "model_evaluation/evaluate.py")
SCORER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCORER)
DATA0_INPUT_SHA = "622410b2918692b75a880b753f1a621e8d892d6dbc13c1f430babc5543a0584e"
BASE = dict(width=48, dropout=.25, lr=.001, weight_decay=.0001,
            pos_weight=3., cap=32, summary_n=63, pool="attention")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def index_hash(indices):
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def source_hashes():
    sources = {name: source_sha256(ROOT / name) for name in
               ("train.py", "model.py", "context_features.py", "data.py", "common.py", "prepare.py")}
    sources["model_evaluation/evaluate.py"] = source_sha256(SCORER.__file__)
    return sources


def atomic_torch(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


class Cache:
    def __init__(self, path):
        path = Path(path)
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest.get("status") != "completed" or manifest.get("input_sha256") != DATA0_INPUT_SHA:
            raise ValueError("Training requires a completed cache of the agreed data0 input")
        for name, expected in manifest["cache_sha256"].items():
            if sha256(path / name) != expected:
                raise ValueError("Changed cache payload: " + name)
        self.keys = pd.read_csv(path / "keys.csv", dtype={"transcript_id": str, "context": str})
        self.off = np.load(path / "offsets.npy", mmap_mode="r", allow_pickle=False)
        self.raw_summary = np.load(path / "summary.npy", mmap_mode="r", allow_pickle=False)
        self.seq = np.load(path / "seq.npy", mmap_mode="r", allow_pickle=False)
        n = len(self.keys)
        if self.off.shape != (n + 1,) or self.off[0] != 0 or np.any(np.diff(self.off) <= 0):
            raise ValueError("Invalid cache offsets")
        if self.raw_summary.shape != (n, 63) or self.seq.shape != (n, 7):
            raise ValueError("Invalid cached summary/sequence shape")
        if not np.isfinite(self.raw_summary).all() or not np.isin(self.seq, range(5)).all():
            raise ValueError("Invalid cached values")
        if not np.array_equal(np.diff(self.off), self.keys.n_reads.to_numpy()):
            raise ValueError("Cached coverage disagrees with offsets")
        if (path / "reads.bin").stat().st_size != int(self.off[-1]) * 9 * 4:
            raise ValueError("Cached reads byte count differs")
        self.reads = np.memmap(path / "reads.bin", dtype="<f4", mode="r", shape=(int(self.off[-1]), 9))
        self.summary = summary_transform(self.raw_summary)
        self.source = sha256(path / "manifest.json")

    def bag(self, index):
        return self.reads[self.off[index]:self.off[index + 1]]


def align_metadata(cache, labels_path, folds_path):
    if sha256(folds_path) != SCORER.FOLD_SHA256:
        raise ValueError("Wrong shared fold fingerprint; restore the exact file")
    labels, genes = SCORER.read_labels(labels_path)
    folds = SCORER.read_folds(folds_path)
    SCORER.same_keys(labels, folds, "Labels/folds")
    if SCORER.canonical_labels_hash(labels, folds) != SCORER.DATA0_LABELS_CANONICAL_SHA256:
        raise ValueError("Changed data0 labels/keys")
    keys = [(str(row.transcript_id), SCORER.integer(str(row.transcript_position), "position"))
            for row in cache.keys.itertuples()]
    if len(set(keys)) != len(keys) or set(keys) != set(labels):
        raise ValueError("Cache and labels require exactly the same unique sites")
    if genes and any(genes[key] != folds[key][0] for key in keys):
        raise ValueError("Label gene mapping differs from shared folds")
    meta = cache.keys.copy()
    meta["label"] = [labels[key] for key in keys]
    meta["gene_id"] = [folds[key][0] for key in keys]
    meta["fold_id"] = [folds[key][1] for key in keys]
    if meta.groupby("gene_id").fold_id.nunique().max() != 1:
        raise ValueError("A gene crosses folds")
    return meta


def choose_population(meta, smoke_genes=None, seed=2026):
    """Select whole genes deterministically without reading labels or scores."""
    if smoke_genes is None:
        return np.arange(len(meta), dtype=np.int64)
    if smoke_genes < 1:
        raise ValueError("smoke_genes must be positive")
    selected = set()
    for fold in range(5):
        genes = set(meta.loc[meta.fold_id == fold, "gene_id"])
        ranked = sorted(genes, key=lambda gene: fingerprint([seed, fold, gene]))
        selected.update(ranked[:smoke_genes])
    return np.flatnonzero(meta.gene_id.isin(selected).to_numpy())


def partition(meta, population, assessment_fold):
    population = np.asarray(population, dtype=np.int64)
    if len(np.unique(population)) != len(population) or np.any((population < 0) | (population >= len(meta))):
        raise ValueError("Invalid population indices")
    if assessment_fold not in range(5):
        raise ValueError("Assessment fold must be 0..4")
    inner_fold = (assessment_fold + 1) % 5
    fold = meta.fold_id.to_numpy()
    assessment = population[fold[population] == assessment_fold]
    inner = population[fold[population] == inner_fold]
    initial = population[~np.isin(fold[population], [assessment_fold, inner_fold])]
    refit = population[fold[population] != assessment_fold]
    groups = {name: set(meta.iloc[idx].gene_id) for name, idx in
              (("initial", initial), ("inner", inner), ("assessment", assessment), ("refit", refit))}
    if not all(len(idx) for idx in (initial, inner, assessment, refit)):
        raise ValueError("Empty gene partition")
    if groups["initial"] & groups["inner"] or groups["refit"] & groups["assessment"]:
        raise ValueError("Gene leakage across fitting/evaluation")
    if groups["initial"] & groups["assessment"] or groups["inner"] & groups["assessment"]:
        raise ValueError("Assessment gene enters inner selection")
    for idx in (initial, inner, refit, assessment):
        if set(meta.iloc[idx].label.unique()) != {0, 1}:
            raise ValueError("Each scored/fitted partition needs both classes; increase smoke gene count if needed")
    audit = {"assessment_fold": assessment_fold, "inner_validation_fold": inner_fold,
             "zero_gene_overlap": True, "populations": {}}
    for name, idx in (("initial_fit", initial), ("inner_validation", inner),
                      ("refit", refit), ("assessment", assessment)):
        sub = meta.iloc[idx]
        audit["populations"][name] = {"sites": len(idx), "genes": sorted(set(sub.gene_id)),
                                     "positives": int(sub.label.sum()), "indices_sha256": index_hash(idx)}
    return initial, inner, refit, assessment, audit


def fit_norm(cache, indices):
    summary = cache.summary[indices].astype(np.float64)
    total = np.zeros(9); squared = np.zeros(9); count = 0
    for start in range(0, len(indices), 1000):
        values = read_transform(np.concatenate([cache.bag(i) for i in indices[start:start + 1000]])).astype(np.float64)
        total += values.sum(0); squared += np.square(values).sum(0); count += len(values)
    mean = total / count
    return dict(read_mean=mean.tolist(), read_sd=np.sqrt(np.maximum(squared / count - mean**2, 1e-8)).tolist(),
                summary_mean=summary.mean(0).tolist(), summary_sd=np.maximum(summary.std(0), 1e-4).tolist(),
                fit_sites=len(indices), fit_reads=count, fit_indices_sha256=index_hash(indices))


def batch(cache, indices, extra, rng=None, cap=None):
    bags = []
    for index in indices:
        reads = cache.bag(index)
        if cap is not None and len(reads) > cap:
            reads = reads[rng.choice(len(reads), cap, replace=False)]
        bags.append(read_transform(reads))
    width = max(map(len, bags))
    reads = np.zeros((len(indices), width, 9), dtype=np.float32)
    mask = np.zeros((len(indices), width), dtype=bool)
    for j, values in enumerate(bags):
        reads[j, :len(values)] = values; mask[j, :len(values)] = True
    tensors = [torch.from_numpy(x) for x in (reads, mask, cache.seq[indices], cache.summary[indices])]
    return (*tensors, None if extra is None else torch.from_numpy(extra[indices]))


@torch.inference_mode()
def predict_cache(model, cache, indices, reference, batch_size=64):
    model.eval()
    extra = None if reference is None else transform_reference(reference, cache.seq, cache.raw_summary)
    result = np.empty(len(indices), dtype=np.float64)
    order = np.argsort(np.diff(cache.off)[indices], kind="stable")
    for start in range(0, len(indices), batch_size):
        at = order[start:start + batch_size]
        result[at] = torch.sigmoid(model(*batch(cache, indices[at], extra))).numpy()
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite predictions")
    return result


def run_fit(cache, y, indices, validation, config, seed, directory, max_epochs,
            patience, batch_size, stage, identity):
    indices = np.asarray(indices, dtype=np.int64)
    if validation is not None:
        validation = np.asarray(validation, dtype=np.int64)
        if np.intersect1d(indices, validation).size:
            raise ValueError("Fitting/validation rows overlap")
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    signature = fingerprint(dict(config=config, seed=seed, max_epochs=max_epochs, patience=patience,
        batch_size=batch_size, stage=stage, identity=identity, threads=torch.get_num_threads(),
        versions=dict(python=platform.python_version(), torch=str(torch.__version__),
                      numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__),
        fitting=index_hash(indices), validation=None if validation is None else index_hash(validation)))
    complete_path, last_path = directory / "complete.json", directory / "last.pt"
    if complete_path.exists():
        completed = json.loads(complete_path.read_text())
        if completed["signature"] != signature or sha256(directory / "best.pt") != completed["best_sha256"]:
            raise ValueError("Completed fit changed; use a new output directory")
        return completed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    with Measure() as measured:
        measured.output = directory / f"invocation_{time.time_ns()}.json"
        state = torch.load(last_path, map_location="cpu", weights_only=False) if last_path.exists() else None
        if state is not None and state.get("signature") != signature:
            raise ValueError("Checkpoint settings/source/split differ; refusing an incompatible resume")
        if state is None:
            norm = fit_norm(cache, indices)
            reference = (fit_reference(cache.seq, cache.raw_summary, indices, min_sites=config["min_sites"])
                         if config["context_mode"] == "relative" else None)
        else:
            norm, reference = state["norm"], state["context_reference"]
            best_bytes = state["best_artifact_bytes"]
            if hashlib.sha256(best_bytes).hexdigest() != state["best_artifact_sha256"]:
                raise ValueError("Checkpoint's retained best artifact is inconsistent")
            # A crash can occur after replacing best.pt but before replacing last.pt.
            # Recover the best artifact committed with this completed epoch first.
            best_path = directory / "best.pt"
            if not best_path.exists() or sha256(best_path) != state["best_artifact_sha256"]:
                temporary = best_path.with_name(best_path.name + ".tmp")
                temporary.write_bytes(best_bytes)
                os.replace(temporary, best_path)
        extra = None if reference is None else transform_reference(reference, cache.seq, cache.raw_summary)
        model = SiteModel(config, norm)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config["pos_weight"]))
        best, best_epoch, stale, history, previous, first = -1., 0, 0, [], [], 1
        if state is not None:
            model.load_state_dict(state["state_dict"]); optimizer.load_state_dict(state["optimizer"])
            torch.set_rng_state(state["torch_rng"])
            random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"])
            best, best_epoch, stale, history = state["best"], state["best_epoch"], state["stale"], state["history"]
            previous, first = state["resources"], state["epoch"] + 1
            if not (directory / "best.pt").exists():
                raise ValueError("Checkpoint's best artifact is missing")
        for epoch in range(first, max_epochs + 1):
            if validation is not None and stale >= patience:
                break
            began = time.perf_counter(); model.train()
            rng = np.random.default_rng(seed * 100000 + epoch)
            shuffled = rng.permutation(indices); loss_sum = 0.
            for start in range(0, len(shuffled), batch_size):
                selected = shuffled[start:start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                logits = model(*batch(cache, selected, extra, rng, config["cap"]))
                loss = criterion(logits, torch.tensor(y[selected], dtype=torch.float32))
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite fitting loss")
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
                loss_sum += float(loss.detach()) * len(selected)
            row = dict(epoch=epoch, loss=loss_sum / len(indices))
            if validation is not None:
                row.update(SCORER.metrics(y[validation], predict_cache(model, cache, validation, reference)))
                selection = row["selection_score"]
            else:
                selection = float(epoch)
            row["wall_seconds"] = time.perf_counter() - began; history.append(row)
            bundle = dict(kind="context_reference_v1", stage=stage, config=config, norm=norm,
                          context_reference=reference, state_dict=model.state_dict(), epoch=epoch,
                          seed=seed, fit_signature=signature, input_fingerprints=identity)
            if selection > best + 1e-5:
                best, best_epoch, stale = selection, epoch, 0
                atomic_torch(directory / "best.pt", bundle)
            else:
                stale += 1
            checkpoint = dict(bundle, signature=signature, optimizer=optimizer.state_dict(), best=best,
                best_epoch=best_epoch, stale=stale, history=history, torch_rng=torch.get_rng_state(),
                python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                best_artifact_bytes=(directory / "best.pt").read_bytes(),
                best_artifact_sha256=sha256(directory / "best.pt"),
                resources=previous + [measured.snapshot()])
            atomic_torch(last_path, checkpoint)
            save_json(directory / "history.json", history)
            (directory / "training.log").write_text("".join(json.dumps(x) + "\n" for x in history), encoding="utf-8")
            print(f"{directory.name}: epoch {epoch}, loss {row['loss']:.5f}"
                  + (f", inner S {selection:.5f}" if validation is not None else ""), flush=True)
        if not history:
            raise ValueError("No completed epoch")
        # Recreate logs even if interruption followed the final checkpoint write.
        save_json(directory / "history.json", history)
        (directory / "training.log").write_text("".join(json.dumps(x) + "\n" for x in history), encoding="utf-8")
    result = dict(signature=signature, config=config, seed=seed, best_epoch=best_epoch,
                  best_inner_score=None if validation is None else best,
                  epochs_completed=len(history), train_sites=len(indices),
                  validation_sites=0 if validation is None else len(validation),
                  fit_indices_sha256=index_hash(indices), resources=previous + [measured.record],
                  best_sha256=sha256(directory / "best.pt"))
    save_json(complete_path, result)
    return result


def restore(path):
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    model = SiteModel(bundle["config"], bundle["norm"])
    model.load_state_dict(bundle["state_dict"]); model.eval()
    return model, bundle


def write_predictions(path, meta, indices, scores):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    frame = meta.iloc[indices][["transcript_id", "transcript_position"]].copy()
    frame["score"] = scores; frame["fold_id"] = meta.iloc[indices].fold_id.to_numpy()
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.17g", lineterminator="\n")
    os.replace(temporary, path)


def verify_saved_fold(directory, fold):
    """Validate every reused fold, including those not requested in this invocation."""
    fold_dir = directory / f"fold{fold}"
    result = json.loads((fold_dir / "fold_result.json").read_text())
    inner = json.loads((fold_dir / "inner/complete.json").read_text())
    refit = json.loads((fold_dir / "refit/complete.json").read_text())
    if (result["refit_signature"] != refit["signature"]
            or result["selected_epochs"] != inner["best_epoch"]
            or result["predictions_sha256"] != sha256(fold_dir / "predictions.csv")
            or refit["best_sha256"] != sha256(fold_dir / "refit/best.pt")
            or inner["best_sha256"] != sha256(fold_dir / "inner/best.pt")):
        raise ValueError(f"Changed completed fold {fold}")
    return result


def run_cv(cache, meta, population, config, args, identity):
    smoke = args.phase == "smoke"
    directory = args.output_dir / ("smoke" if smoke else "cv") / config["context_mode"]
    directory.mkdir(parents=True, exist_ok=True)
    recipe = dict(config=config, max_epochs=args.max_epochs, patience=args.patience,
                  seed=args.seed, batch_size=args.batch_size, threads=args.threads,
                  population_sha256=index_hash(population), identity=identity, smoke=smoke)
    recipe_path = directory / "recipe.json"
    if recipe_path.exists() and json.loads(recipe_path.read_text()) != recipe:
        raise ValueError("Changed CV recipe; choose a new output directory")
    save_json(recipe_path, recipe)
    y = meta.label.to_numpy()
    for fold in args.outer_folds:
        initial, validation, refit, assessment, audit = partition(meta, population, fold)
        fold_dir = directory / f"fold{fold}"
        save_json(fold_dir / "split_audit.json", audit)
        tuned = run_fit(cache, y, initial, validation, config, args.seed + fold, fold_dir / "inner",
                        args.max_epochs, args.patience, args.batch_size,
                        "smoke" if smoke else "cv_inner", identity)
        fitted = run_fit(cache, y, refit, None, config, args.seed + 100 + fold, fold_dir / "refit",
                         tuned["best_epoch"], args.patience, args.batch_size,
                         "smoke" if smoke else "cv_refit", identity)
        result_path, prediction_path = fold_dir / "fold_result.json", fold_dir / "predictions.csv"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result["refit_signature"] != fitted["signature"] or sha256(prediction_path) != result["predictions_sha256"]:
                raise ValueError("Changed completed fold result")
            continue
        model, bundle = restore(fold_dir / "refit/best.pt")
        with Measure() as measured:
            scores = predict_cache(model, cache, assessment, bundle["context_reference"])
        write_predictions(prediction_path, meta, assessment, scores)
        save_json(result_path, dict(fold_id=fold, selected_epochs=tuned["best_epoch"],
            refit_signature=fitted["signature"], predictions_sha256=sha256(prediction_path),
            metrics=SCORER.metrics(y[assessment], scores), inference_resources=measured.record,
            interpretation="pipeline smoke check only" if smoke else "gene-fold development result"))
    available = [f for f in range(5) if (directory / f"fold{f}/fold_result.json").exists()]
    for fold in available:
        verify_saved_fold(directory, fold)
    progress = dict(completed_folds=available, full_five_folds=len(available) == 5,
                    smoke=smoke, full_data0_population=not smoke)
    save_json(directory / "progress.json", progress)
    if not smoke and len(available) == 5:
        frames = [pd.read_csv(directory / f"fold{f}/predictions.csv", dtype={"transcript_id": str},
                              float_precision="round_trip") for f in range(5)]
        combined = pd.concat(frames).sort_values(["transcript_id", "transcript_position"], kind="stable")
        temporary = directory / "data0_oof.csv.tmp"
        combined.to_csv(temporary, index=False, float_format="%.17g", lineterminator="\n")
        os.replace(temporary, directory / "data0_oof.csv")
        result = SCORER.evaluate_files(args.labels, directory / "data0_oof.csv", "data0", folds_path=args.folds)
        for row in result["per_fold"]:
            saved = verify_saved_fold(directory, row["fold_id"])["metrics"]
            if any(abs(float(row[key]) - float(value)) > 1e-12 for key, value in saved.items()):
                raise ValueError("Merged OOF metrics differ from a saved fold result")
        SCORER.write_json(directory / "data0_cv_metrics.json", result)
        progress.update(oof_sha256=sha256(directory / "data0_oof.csv"),
                        cv_metrics_sha256=sha256(directory / "data0_cv_metrics.json"))
        save_json(directory / "progress.json", progress)


def run_final(cache, meta, config, args, identity):
    cv_directory = args.output_dir / "cv" / config["context_mode"]
    recipe = json.loads((cv_directory / "recipe.json").read_text())
    progress = json.loads((cv_directory / "progress.json").read_text())
    if not progress["full_five_folds"] or not progress["full_data0_population"]:
        raise ValueError("Final fit requires completed full-population five-fold CV")
    if (recipe["config"] != config or recipe["identity"] != identity
            or recipe["seed"] != args.seed or recipe["batch_size"] != args.batch_size
            or recipe["threads"] != args.threads):
        raise ValueError("Final recipe differs from the evaluated candidate")
    if not (cv_directory / "data0_cv_metrics.json").exists():
        raise ValueError("Missing shared CV evaluation")
    if (progress.get("oof_sha256") != sha256(cv_directory / "data0_oof.csv")
            or progress.get("cv_metrics_sha256") != sha256(cv_directory / "data0_cv_metrics.json")):
        raise ValueError("Completed CV evidence changed")
    epochs = int(np.median([verify_saved_fold(cv_directory, f)["selected_epochs"] for f in range(5)]))
    final_directory = args.output_dir / "final" / config["context_mode"]
    frozen = dict(config=config, identity=identity, seed=args.seed, batch_size=args.batch_size,
                  threads=args.threads, selected_epochs=epochs,
                  selection_rule="median of five inner-selected epoch counts",
                  cv_metrics_sha256=sha256(cv_directory / "data0_cv_metrics.json"))
    frozen_path = final_directory / "frozen_specification.json"
    if frozen_path.exists() and json.loads(frozen_path.read_text()) != frozen:
        raise ValueError("Frozen final specification changed; use a new output directory")
    save_json(frozen_path, frozen)
    completed = run_fit(cache, meta.label.to_numpy(), np.arange(len(meta)), None, config, args.seed,
                        final_directory, epochs, args.patience,
                        args.batch_size, "final_fitting", identity)
    if completed["epochs_completed"] != epochs or sha256(final_directory / "best.pt") != completed["best_sha256"]:
        raise ValueError("Final fit is incomplete or changed")
    bundle = torch.load(final_directory / "best.pt", map_location="cpu", weights_only=False)
    bundle["stage"] = "final_data0"
    bundle["fit_completed"] = dict(signature=completed["signature"],
        best_artifact_sha256=completed["best_sha256"], epochs_completed=completed["epochs_completed"],
        selected_epochs=epochs)
    model_path = final_directory / "model.pt"
    if not model_path.exists():
        atomic_torch(model_path, bundle)
    else:
        frozen_model_path = final_directory / "frozen_model.json"
        if frozen_model_path.exists() and json.loads(frozen_model_path.read_text())["model_sha256"] != sha256(model_path):
            raise ValueError("Saved frozen model changed")
        existing = torch.load(model_path, map_location="cpu", weights_only=False)
        if (existing.get("fit_completed") != bundle["fit_completed"]
                or any(existing.get(key) != bundle[key] for key in
                       ("kind", "stage", "config", "norm", "context_reference", "input_fingerprints"))
                or set(existing.get("state_dict", {})) != set(bundle["state_dict"])
                or any(not torch.equal(existing["state_dict"][key], value)
                       for key, value in bundle["state_dict"].items())):
            raise ValueError("Existing final model differs from the completed fit")
    save_json(final_directory / "frozen_model.json", dict(model_sha256=sha256(model_path),
        specification_sha256=sha256(frozen_path), fit_completed=bundle["fit_completed"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--folds", type=Path, default=SCORER.DEFAULT_FOLDS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=["smoke", "cv", "final"], default="smoke")
    parser.add_argument("--mode", choices=["none", "relative", "both"], default="both")
    parser.add_argument("--outer-folds", default="0,1,2,3,4")
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-context-sites", type=int, default=30)
    parser.add_argument("--smoke-genes-per-fold", type=int, default=20)
    args = parser.parse_args(argv)
    args.outer_folds = [int(x) for x in args.outer_folds.split(",")]
    if not args.outer_folds or len(set(args.outer_folds)) != len(args.outer_folds) or not set(args.outer_folds) <= set(range(5)):
        parser.error("--outer-folds must be distinct IDs in 0..4")
    if min(args.max_epochs, args.patience, args.batch_size, args.threads, args.min_context_sites) < 1 or args.seed < 0:
        parser.error("Counts must be positive and seed nonnegative")
    if args.phase == "final" and args.mode == "both":
        parser.error("Nominate one --mode for the final saved model")
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = Cache(args.cache); meta = align_metadata(cache, args.labels, args.folds)
    identity = dict(cache_manifest_sha256=cache.source, data0_input_sha256=DATA0_INPUT_SHA,
                    labels_sha256=sha256(args.labels), folds_sha256=sha256(args.folds),
                    source_sha256=source_hashes())
    population = choose_population(meta, args.smoke_genes_per_fold if args.phase == "smoke" else None, args.seed)
    invocation = dict(phase=args.phase, modes=args.mode, outer_folds=args.outer_folds, sites=len(population),
                      full_data0_sites=len(meta), genes=int(meta.iloc[population].gene_id.nunique()),
                      identity=identity, python=platform.python_version(), torch=str(torch.__version__),
                      arguments=list(sys.argv[1:] if argv is None else argv),
                      command_template="python train.py --cache DATA0_CACHE --labels DATA0_LABELS --output-dir OUTPUT [options]",
                      interpretation="software smoke check, not model comparison" if args.phase == "smoke" else "data0-only model development")
    save_json(args.output_dir / f"invocation_{time.time_ns()}.json", invocation)
    for mode in (["none", "relative"] if args.mode == "both" else [args.mode]):
        config = dict(BASE, context_mode=mode, min_sites=args.min_context_sites)
        if args.phase == "final":
            run_final(cache, meta, config, args, identity)
        else:
            run_cv(cache, meta, population, config, args, identity)
    print("Completed requested stage. data1 was not read; smoke results are not final model evidence.", flush=True)


if __name__ == "__main__":
    main()
