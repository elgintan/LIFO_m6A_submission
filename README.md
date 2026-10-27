# LIFO: m6A site prediction

This release contains the selected CPU model, prediction code, fixed-recipe
training code and a small runnable example for the DSA4262 code submission.
Run commands from the repository root. Prediction uses the included model;
training is not required before making predictions.

## 1. Install on Ubuntu

Use Python 3.12 in an isolated environment. The following commands use uv to
install that Python version even when the system Python has a different version:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
"$HOME/.local/bin/uv" python install 3.12
"$HOME/.local/bin/uv" venv --python 3.12 --seed .venv
source .venv/bin/activate
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r model_elgin/context_reference_v1/requirements.txt
python -m pip check
```

These commands need network access. If curl or CA certificates are missing,
install the Ubuntu packages `curl` and `ca-certificates` first. Install the CPU
Torch wheel before the requirements file to avoid downloading a CUDA runtime.
The predictor uses CPU and does not require a GPU, Docker or RONIN software.
The same model and pinned dependencies passed prediction checks on Ubuntu 24.04
and native AWS Ubuntu 26.04 with Python 3.12.15.

## 2. Run the small prediction example

```bash
python -B model_elgin/context_reference_v1/predict.py --input model_elgin/artifacts/cap128_weight3_v1/example/input.json --model model_elgin/artifacts/cap128_weight3_v1/model.pt --output example_predictions.csv --threads 4
```

Expected terminal result: `PASS: 10 sites / 1,887 reads, stage final_data0,
all-read prediction`. The CSV has ten rows after this exact header:

```csv
transcript_id,transcript_position,score
```

Expected keys and scores are supplied in
`model_elgin/artifacts/cap128_weight3_v1/example/expected.csv`. Allow an absolute
score difference of 0.000003 for platform-dependent floating-point arithmetic.
The companion `example_predictions.csv.manifest.json` records input/model/code
fingerprints, exact key coverage and use of all available reads. The example
contains ten real data0 training sites; it checks software execution and format,
not accuracy on unseen data.

## 3. Predict another compatible input

Replace the input and output paths in the example command. Input must follow
the course's nine-measurement site JSON format; plain JSON and JSON.gz are
supported. Labels are not required. Each input site appears once in the CSV,
with its transcript identifier and position preserved and a continuous score
between 0 and 1. No read-count filter or binary cutoff is applied. The model
uses all valid reads at prediction time, even though training sampled up to
128 reads per site.

Keep the manifest beside the CSV. The script verifies input and output contents
and rejects malformed inputs or incompatible existing output files. Use a new
output filename for a different input or model. A score is a site prediction,
not a measured fraction of modified RNA molecules.

The selected model is the original cap128, positive-weight3 neural model,
fitted for 18 epochs on data0 only. Its SHA256 is:

```text
b2d38ff29a918cb6213ad7df2c76786c5f56d3640bd055ff179d211b0e979668
```

## 4. Reproduce the fixed final training recipe

Obtain the original course data0 input and labels separately. They are not
redistributed here. In the command below, replace the two `LOCAL_DATA` paths
with the locations of those original files:

```bash
python -B -u model_elgin/train_final.py --input LOCAL_DATA/data0/dataset0.json.gz --labels LOCAL_DATA/data0/data.info.labelled --output-dir model_elgin/outputs/reproduced_final
```

The command validates the original input/label fingerprints, prepares the
label-free read cache, and fits fresh normalisers and model weights for the
fixed 18 epochs using seed 2026, batch size 256 and four CPU threads. Settings
are recorded in `model_elgin/artifacts/cap128_weight3_v1/training_specification.json`.
No private historical model weights, CV checkpoints or experiment logs are
required. The shared fold CSV and scoring helper are included because the
training implementation imports and checks them.

Use a fresh output directory for a new reproduction. The resulting `model.pt`
is written inside that directory. Repeating the same command resumes completed
epochs after an interruption. Do not remove `run.lock` while its original process
is running. `--check-only` validates/prepares the input without fitting; an
existing verified cache may be supplied with `--cache`.

This command reproduces the fixed final recipe. It does not retune the model
or replay the whole development comparison. A separately reproduced artifact
has its own provenance fingerprint; use the included nominated artifact when
reproducing this release's predictions. No small training example is required.

## Files

- `model_elgin/context_reference_v1/`: parsing, model, prediction and training code.
- `model_elgin/train_final.py`: fixed final-recipe training entry point.
- `model_elgin/artifacts/cap128_weight3_v1/`: saved model, recipe, metadata and example.
- `model_evaluation/evaluate.py`: scorer required by the training implementation.
- `eda/received/shared_folds/data0_folds.csv`: original shared gene-fold assignments.
- `CODE_RELEASE_MANIFEST.json`: exact file fingerprints for this code release.
