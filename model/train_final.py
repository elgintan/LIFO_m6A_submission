"""Train the published fixed final recipe from raw data0 and labels.

This reproduces the frozen 18-epoch recipe without requiring private historical
CV checkpoints. It does not retune or independently replay the five-fold study.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "context_reference_v1"))
import train as core
from data import prepare, sha256
from model import source_sha256


def promote(directory, completed):
    """Same promotion contract as core.run_final, without historical CV files."""
    if completed["epochs_completed"] != 18 or completed["best_epoch"] != 18 or completed["validation_sites"] != 0:
        raise ValueError("Incomplete prescribed final fit")
    best = directory / "best.pt"
    if sha256(best) != completed["best_sha256"]:
        raise ValueError("Completed checkpoint changed")
    bundle = core.torch.load(best, map_location="cpu", weights_only=False)
    bundle["stage"] = "final_data0"
    bundle["fit_completed"] = dict(signature=completed["signature"], best_artifact_sha256=completed["best_sha256"],
                                   epochs_completed=18, selected_epochs=18)
    model = directory.parent / "model.pt"
    receipt = directory.parent / "frozen_model.json"
    if model.exists():
        if receipt.exists() and sha256(model) != json.loads(receipt.read_text())["model_sha256"]:
            raise ValueError("Existing final artifact changed")
        existing = core.torch.load(model, map_location="cpu", weights_only=False)
        if (existing.get("fit_completed") != bundle["fit_completed"]
                or any(existing.get(k) != bundle[k] for k in ("kind", "stage", "config", "norm", "input_fingerprints"))
                or set(existing.get("state_dict", {})) != set(bundle["state_dict"])
                or any(not core.torch.equal(existing["state_dict"][k], v) for k, v in bundle["state_dict"].items())):
            raise ValueError("Existing artifact differs from this completed fit")
    else:
        core.atomic_torch(model, bundle)
    core.save_json(receipt, dict(model_sha256=sha256(model), fit_completed=bundle["fit_completed"]))
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Original dataset0.json.gz")
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="New local run directory")
    parser.add_argument("--cache", type=Path, help="Optional existing validated data0 cache; read only")
    parser.add_argument("--check-only", action="store_true", help="Validate/prepare without model fitting")
    args = parser.parse_args()
    specification = HERE / "artifacts/cap128_weight3_v1/training_specification.json"
    spec = json.loads(specification.read_text(encoding="utf-8"))
    expected = dict(core.BASE, cap=128, context_mode="none", min_sites=30)
    if (spec["config"] != expected or spec["selected_epochs"] != 18 or spec["seed"] != 2026
            or spec["batch_size"] != 256 or spec["threads"] != 4):
        raise ValueError("Changed published final recipe")
    if sha256(args.input) != spec["identity"]["data0_input_sha256"] or sha256(args.labels) != spec["identity"]["labels_sha256"]:
        raise ValueError("This frozen project recipe trains only on the agreed data0")
    for name, digest in core.source_hashes().items():
        if spec["identity"]["source_sha256"][name] != digest:
            raise ValueError("Frozen core source changed: " + name)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not (output / "reproduction_specification.json").exists():
        raise ValueError("Use a new run directory; unrelated existing files are preserved")
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "run.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write("Original execution must end before clearing this lock.\n")
    try:
        cache_path = args.cache or output / "cache"
        if args.cache is None:
            prepare(args.input, cache_path)
        cache = core.Cache(cache_path)
        folds = HERE.parent / "eda/received/shared_folds/data0_folds.csv"
        meta = core.align_metadata(cache, args.labels, folds)
        if len(meta) != 121838 or int(meta.label.sum()) != 5475 or int(meta.gene_id.nunique()) != 3852:
            raise ValueError("Wrong full data0 population")
        # Fresh cache manifests contain local paths: record the actual new hash.
        # This changes provenance, never the frozen measurements or algorithm.
        identity = dict(spec["identity"], cache_manifest_sha256=cache.source,
                        final_reproduction_entrypoint_sha256=source_sha256(__file__),
                        published_specification_sha256=sha256(specification))
        frozen = dict(config=expected, identity=identity, epochs=18, seed=2026, batch_size=256, threads=4,
                      scope="fixed all-data0 reproduction; not new CV/tuning")
        path = output / "reproduction_specification.json"
        if path.exists() and json.loads(path.read_text()) != frozen:
            raise ValueError("Settings/source/input changed; use a new output directory")
        core.save_json(path, frozen)
        if args.check_only:
            print("PASS: complete data0/cache/labels/source/frozen recipe; no fitting", flush=True)
            return
        core.torch.set_num_threads(4)
        with core.Measure() as measured:
            completed = core.run_fit(cache, meta.label.to_numpy(), np.arange(len(meta)), None, expected,
                                     2026, output / "fit", 18, 5, 256, "final_fitting", identity)
            model = promote(output / "fit", completed)
        if source_sha256(__file__) != identity["final_reproduction_entrypoint_sha256"]:
            raise ValueError("Training entrypoint changed during execution")
        current_sources = core.source_hashes()
        if any(current_sources[name] != value for name, value in spec["identity"]["source_sha256"].items()
               if name in current_sources):
            raise ValueError("Core changed during execution")
        core.save_json(output / "complete.json", dict(status="complete", model_sha256=sha256(model),
                       resources=measured.record, no_data1_fitting=True, no_new_validation_claim=True))
        print("PASS: fresh fixed-recipe model saved; use original OOF results for validation", flush=True)
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
