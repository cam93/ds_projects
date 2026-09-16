# Behavioral features and reproducible model comparison

The comparison addresses the weak expanded-data candidate without relaxing its production release
policy. It adds richer inputs, compares three model families, and records a justified selection
with scenario-level results. The comparison policy is explicitly for a synthetic demo; it does
not approve a model for production. The existing deployed artifact remains unchanged.

## Features and ingestion

`src/fraud_detector/features/handbook.py` is shared by batch preparation and live simulation.
Version 1 has amount, hourly transaction count, customer history duration and UTC hour. Version 2
adds 14 values: five-minute and daily counts; prior customer count, mean amount and standard
deviation; relative amount and amount z-score; terminal novelty and prior customer-terminal visits;
terminal hourly count and mean amount; relative terminal amount; and cyclic hour sine/cosine.

Every historical value excludes the current transaction. Events are ordered by timestamp and
lexical transaction ID; equal-time events follow that order and window boundaries are inclusive.
Late events are rejected. No fraud label, scenario or customer/terminal identifier is a model input.
Training-only normalization prevents future statistics entering training. History may carry across
split boundaries because those unlabeled transactions would already have occurred at scoring time.

For a new customer/terminal, means and standard deviations start at zero, amount ratios at one,
and the z-score at zero until two prior purchases exist. The z-score uses a 0.01 denominator floor
and clips to ±1000. Lifetime aggregates require retained state; the demo journal grows with traffic.
This local simulator state is not a distributed production feature store.

The API accepts the additional values in a complete, validated `behavioral_features` object.
Legacy four-feature requests/artifacts still work. An 18-feature model returns HTTP 422 if this
block is missing. Producers remain authenticated and trusted: the API validates consistency but
cannot prove historical values without the source history. Existing legacy idempotency hashes
are preserved. Simulator checkpoints migrate automatically without changing pending request bodies.

## Reproduce the evaluation

Use Python 3.10 and the project's virtual environment:

```bash
.venv/bin/python -m pip install -c constraints.txt -e '.[training]'
.venv/bin/python scripts/simulate.py batch --seed 7 --days 90 --start 2024-01-01 \
  --output data/raw/comparison-training.csv
.venv/bin/python scripts/prepare_dataset.py data/raw/comparison-training.csv \
  --output-dir data/processed/comparison-training
.venv/bin/python scripts/simulate.py batch --seed 29 --days 90 --start 2024-04-01 \
  --output data/raw/comparison-independent.csv
.venv/bin/python scripts/prepare_dataset.py data/raw/comparison-independent.csv \
  --output-dir data/processed/comparison-independent
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/compare_models.py \
  --input data/processed/comparison-training/training_features.csv \
  --external-input data/processed/comparison-independent/training_features.csv \
  --policy configs/model-comparison.json --output-dir reports/comparison-new-run
```

Choose new output names; generation and comparison refuse to overwrite existing runs. For the
completed run, local feature files are in `data/processed/simulated-demo-v2` and
`data/processed/simulated-independent-v2`. Raw data manifests record the generator configuration.
The input hashes, dependency versions and model settings are retained in the comparison report.

The first population is split chronologically 70%/15%/15%, keeping tied timestamps together.
Each partition must contain at least 100 fraud and 1,000 legitimate examples. All six candidates
use the same partitions: logistic regression, histogram gradient boosting and an MLP, each with
four or 18 features. Logistic/tree settings are held fixed; MLP width grows from 8 to 32, so its
comparison reflects both features and capacity. Seeds and CPU thread counts are fixed.

Each threshold maximizes validation recall while keeping validation false positives at or below
1% of legitimate transactions. Score ties stay together. Selection uses validation recall,
then precision, then average precision. This is a declared demo operating point, not a guarantee
that test or live false positives stay under 1%. No candidate is chosen using test metrics.

`selection-before-test.json` freezes the choice before test/external scoring. All six candidates
are then evaluated without refitting on the chronological test partition and seed-29 population.
The latter starts with fresh customer histories and includes its cold-start period. A separate
seed tests another population from the same generator, not generalization to real banking data.
The original test period has already appeared in an earlier candidate report and is not pristine.

## Outputs and deployment boundary

- `MODEL_SELECTION.md`: readable comparison, scenario recall and limitations.
- `comparison.json`: confusion counts, precision/recall, FPR, average precision, ROC AUC,
  mean daily precision@100, per-scenario recall, settings and provenance.
- `*-test-predictions.csv`: local row-level audit files; excluded from Git for the completed run.
- `selected-model.json` and `.sha256`: data-only portable model with checked native-score parity.

Runtime supports this JSON format without importing scikit-learn or deserializing executable
pickle objects. The artifact explicitly has `release.approved=false`; production refuses it.
The legacy `scripts/train.py` workflow still trains and gates four-feature PyTorch artifacts.
Do not overwrite `fraud_model.pt` with JSON or mark a candidate approved to bypass evaluation.
To trial a reviewed JSON candidate in an isolated development service, configure `MODEL_PATH`
to its mounted `.json` path and `MODEL_SHA256` to its sidecar, and send v2-enriched requests.
Changing model paths/mounts and promoting the demo deployment are separate deliberate operations.

To verify the candidate in disposable containers without changing the deployed stack:

```bash
docker build -f deploy/docker/api.Dockerfile -t fraud-detector-review:local .
.venv/bin/python scripts/smoke_compose.py --model reports/model-comparison/selected-model.json
```

This uses temporary credentials, a separate Compose project and no published ports. It checks
replay, live simulation across a restart, metrics and anonymous Viewer permissions, then removes
its containers and volumes. Omit `--model` to verify the legacy four-feature artifact path.

Compromised-terminal fraud is partly indistinguishable from genuine purchases in this generator.
The report makes those missed cases visible rather than encoding secret fraud labels in features.
Class weighting also means model scores are not calibrated fraud probabilities. Real deployment
still needs representative evaluation, an agreed operating policy and a durable shared feature
service appropriate for multiple producers.
