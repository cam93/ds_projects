# Realtime Fraud Detector

An end-to-end realtime financial fraud detection platform built with PyTorch,
Docker, Terraform, Prometheus, and Grafana.

## Five-container runtime

Terraform provisions exactly five containers:

1. `traffic-generator` continuously posts synthetic transactions.
2. `api` validates events, scales features, runs the PyTorch scorer, exposes
   Prometheus metrics, and sends every prediction to the audit service.
3. `audit-store` persists predictions in a SQLite database.
4. `prometheus` scrapes the API metrics endpoint.
5. `grafana` displays provisioned live prediction, fraud-rate, and latency
   charts.

## Repository layout

```text
.
├── apps/
│   ├── audit_store/            # SQLite-backed prediction audit API
│   └── traffic_generator/      # Continuous synthetic transaction producer
├── configs/                    # Shared application and model configuration
├── data/
│   ├── raw/                    # Source datasets (not committed)
│   ├── interim/                # Validated/intermediate datasets
│   └── processed/              # Training-ready datasets
├── deploy/
│   ├── docker/                 # Service Dockerfiles
│   └── kubernetes/             # Optional Kubernetes manifests
├── docs/                       # Architecture and operational documentation
├── infra/terraform/
│   ├── environments/           # dev/staging/prod root modules
│   └── modules/                # Reusable cloud infrastructure modules
├── monitoring/                 # Prometheus and Grafana configuration
├── notebooks/                  # Exploratory analysis and experiments
├── scripts/                    # Data, training, and local operations scripts
├── src/fraud_detector/         # Shared Python package
├── tests/                      # Unit, integration, and contract tests
├── docker-compose.yml          # Local five-container development stack
├── pyproject.toml              # Python tooling and dependencies
└── .env.example                # Local configuration template
```

## Getting started

1. Copy `.env.example` to `.env` and adjust local values.
2. Start the five-container local stack with `docker compose up --build`.
3. Open the API at `http://localhost:8000/docs`, Prometheus at
   `http://localhost:9090`, and Grafana at `http://localhost:3000`.
4. To provision the same five-container topology with Terraform:
   `cd infra/terraform/environments/dev && terraform init && terraform apply`.

Terraform resolves the project root to an absolute path for Docker bind mounts.
The root `.dockerignore` excludes local virtual environments, Terraform state,
and generated caches so Docker image builds remain small and reliable.
Grafana mounts only its provisioning and dashboard directories, preserving the
image's built-in configuration and startup paths.

Before starting the Compose stack, prepare the replay file:

```bash
.venv/bin/python scripts/prepare_dataset.py data/raw/*.pkl \
  --output-dir data/processed
```

The traffic generator runs in `handbook` mode by default, replays
`data/processed/handbook/replay.jsonl` at four events per second, preserves
source timestamps, and prefixes transaction IDs with the configured replay run
ID (for example, `run001:123`). The original source ID is retained for audit
and evaluation mapping.

## Preparing the supplied dataset

The ingestion pipeline reads a pandas pickle, validates required transaction
IDs, timestamps, and positive amounts, then sorts by timestamp and transaction
ID. It writes three deterministic artifacts:

```bash
.venv/bin/python scripts/prepare_dataset.py \
  data/raw/2018-04-01.pkl \
  --output-dir data/processed
```

Multiple pickle files can be supplied together; validation and sorting are
performed across the combined dataset:

```bash
.venv/bin/python scripts/prepare_dataset.py data/raw/*.pkl \
  --output-dir data/processed
```

The shared feature handbook in
[`src/fraud_detector/features/handbook.py`](/Users/cameron/DS_projects/ds_projects/financial_anomaly_/src/fraud_detector/features/handbook.py)
calculates `amount`, `transactions_last_hour`, `customer_history_days`, and
`hour_of_day` point-in-time. Customer history state is retained while multiple
daily files are processed, so replay does not reset at a file boundary.

## Training the model

Train the classifier after preparing the feature table:

```bash
.venv/bin/python scripts/train.py \
  --input data/processed/training_features.csv \
  --output models/artifacts/fraud_model.pt
```

Training uses chronological 70/15/15 train, validation, and test periods.
Normalization parameters are fitted only on the training period. The exported
artifact contains model weights, feature order, normalization parameters,
validation-selected threshold, model version, and test metrics. The API loads
this artifact at prediction time rather than using hard-coded weights or a
fixed decision threshold.

- `records.csv`: validated source records
- `training_features.csv`: numeric training table with fraud labels
- `replay_events.jsonl`: API-compatible events in replay order

Place additional source pickle files in `data/raw/`; each file can be prepared
with the same command and its outputs written to a separate processed
directory.

The Compose project name and locally built image tags are explicitly pinned, so
the stack can also be run from directories whose names contain underscores or
trailing separators.

The initial tree is intentionally scaffolded so model architecture, event
transport, persistence, and cloud-provider-specific Terraform can be added
without moving public interfaces.

## End-to-end data and prediction flow

The platform follows a deterministic pipeline from raw transaction data to
live predictions and audit trail:

```
┌─────────────────────────────────────────────────────────────────┐
│ Handbook Dataset (pickle)                                        │
│   - Raw transactions with labels, amounts, timestamps            │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ├────────────────────────────────────────┐
                     v                                        │
        ┌────────────────────────────┐                        │
        │ 1. Data Preparation        │                        │
        │  (prepare_dataset.py)      │                        │
        ├────────────────────────────┤                        │
        │ ✓ Validate records         │                        │
        │ ✓ Sort by timestamp + ID   │                        │
        │ ✓ Calculate features       │                        │
        │ ✓ Split chronologically    │                        │
        └──┬───────────────────────┬─┘                        │
           │                       │                          │
           ├─→ records.csv         │                          │
           └─→ training_features.csv                          │
                                   │                          │
        ┌──────────────────────────┴─────┐                    │
        │  2a. Model Training            │                    │
        │  (train.py)                    │                    │
        ├────────────────────────────────┤                    │
        │ Train (70%):                   │                    │
        │  ├─ Fit normalization          │                    │
        │  └─ Train PyTorch MLP          │                    │
        │                                │                    │
        │ Validation (15%):              │                    │
        │  └─ Threshold selection (F1)   │                    │
        │                                │                    │
        │ Test (15%):                    │                    │
        │  └─ Precision, Recall, AP      │                    │
        └──┬─────────────────────────────┘                    │
           │                                                  │
           v                                                  │
        ┌───────────────────────────────┐                     │
        │ fraud_model.pt (artifact)     │                     │
        │ - Model weights               │                     │
        │ - Normalization params        │                     │
        │ - Decision threshold          │                     │
        │ - Version info                │                     │
        └────────────────────────────────┘                    │
                                                              │
        ┌──────────────────────────────────┐                 │
        │ 2b. Replay Events Preparation   │◄────────────────┘
        │ (prepare_dataset.py)             │
        ├──────────────────────────────────┤
        │ ✓ Test set → JSONL format        │
        │ ✓ Preserve source transaction_id │
        │ ✓ Point-in-time features         │
        └──┬───────────────────────────────┘
           │
           v
        ┌─────────────────────────┐
        │ replay.jsonl (9,488     │
        │ test transactions)      │
        └─────────┬───────────────┘
                  │
                  │ docker compose up
                  ├─────────────────────────────────────────┐
                  │                                         │
        ┌─────────v──────────────┐              ┌──────────v──────────────┐
        │ 3. Traffic Generator   │              │ 4. Fraud Detection API │
        │ (traffic_generator.py) │─HTTP POST──→ │ (api.py)               │
        ├────────────────────────┤              ├───────────────────────┤
        │ Replay mode:           │              │ ✓ Parse transaction  │
        │ ├─ Read replay.jsonl   │              │ ✓ Calculate features │
        │ ├─ Post @4 events/sec  │              │ ✓ Apply normalization│
        │ ├─ Prefix IDs with run_id              │ ✓ Score with model   │
        │ └─ Preserve timestamps │              │ ✓ Apply threshold    │
        │                        │              │ ✓ Emit Prometheus    │
        │ (run_id=run001)        │              │ ✓ Send to audit store│
        └────────────────────────┘              └──────────┬────────────┘
                                                          │
                                        ┌─────────────────┴────────────────┐
                                        │                                  │
                                        v                                  v
                    ┌────────────────────────────┐       ┌──────────────────────┐
                    │ 5. Audit Store (SQLite)    │       │ 6. Prometheus        │
                    ├────────────────────────────┤       ├──────────────────────┤
                    │ Every prediction logged:   │       │ ✓ Prediction count   │
                    │ ├─ transaction_id          │       │ ✓ Fraud rate         │
                    │ ├─ source_transaction_id   │       │ ✓ Prediction latency │
                    │ ├─ timestamp               │       │ ✓ Model version      │
                    │ ├─ fraud_probability       │       └──────────┬───────────┘
                    │ ├─ is_fraud                │                  │
                    │ └─ model_version           │                  v
                    └────────────────────────────┘       ┌────────────────────────┐
                                                         │ 7. Grafana Dashboards  │
                                                         ├────────────────────────┤
                                                         │ ✓ Fraud rate timeline  │
                                                         │ ✓ Prediction latency   │
                                                         │ ✓ Model version track  │
                                                         │ ✓ Live metrics         │
                                                         └────────────────────────┘
```

### 1. Dataset Preparation (`data/processed/`)

Ingests raw pickle files, validates unique transaction IDs and positive
amounts, sorts chronologically, and outputs three deterministic artifacts:

- **`records.csv`**: validated source transactions
- **`training_features.csv`**: numerical feature table with labels
- **`handbook/replay.jsonl`**: API-compatible test events in order

Feature calculation uses [`handbook.py`](/Users/cameron/DS_projects/ds_projects/financial_anomaly_/src/fraud_detector/features/handbook.py):
all features are computed **before** adding the current transaction to the
customer history state. This ensures point-in-time consistency across training
and inference.

### 2. Model Training (`models/artifacts/fraud_model.pt`)

Training pipeline:

- **Chronological split**: 70% training, 15% validation, 15% test
- **Normalization**: fitted **only** on training period to prevent leakage
- **Model**: 2-layer PyTorch MLP with BCE loss and `pos_weight` for imbalance
- **Threshold selection**: selected on validation period to maximize F1 score
- **Artifact export**: weights, normalization params, threshold, version

### 3. Replay and Live Inference

- **Traffic generator** reads `replay.jsonl`, posts test transactions at 4
  events/second.
- **ID prefixing**: transaction IDs are prefixed with run ID (e.g.,
  `run001:123`) to allow multiple replays without audit duplicates.
- **Source preservation**: original `source_transaction_id` is retained for
  evaluation mapping back to test labels.

### 4. Audit and Evaluation

Every prediction is persisted with:

- `transaction_id`: prefixed ID (e.g., `run001:123`)
- `source_transaction_id`: original ID (e.g., `123`) for evaluation join
- `fraud_probability`: model output
- `is_fraud`: thresholded decision
- `model_version`: artifact version for multimodel tracking

Join predictions back to held-out test labels using `source_transaction_id` to
compute precision, recall, and average precision on unseen data.

## Running the complete flow

### Step 1: Prepare the dataset

```bash
.venv/bin/python scripts/prepare_dataset.py data/raw/2018-04-01.pkl \
  --output-dir data/processed
```

Validates 9,488 transactions, creates deterministic artifacts.

### Step 2: Train the model

```bash
.venv/bin/python scripts/train.py \
  --input data/processed/training_features.csv \
  --output models/artifacts/fraud_model.pt
```

Exports model weights, normalization, threshold, and version.

### Step 3: Start the runtime stack

```bash
docker compose up --build
```

Launches five containers:
- API listening at `http://localhost:8000`
- Audit store at `http://localhost:8001`
- Prometheus at `http://localhost:9090`
- Grafana at `http://localhost:3000`
- Traffic generator replaying 9,488 test transactions

### Step 4: Monitor

Open Grafana at `http://localhost:3000`:
- **Fraud Detection Rate**: live fraud probability distribution
- **Prediction Latency**: API response time histogram
- **Model Version**: currently deployed artifact version

Open `http://localhost:8001/predictions` to inspect the audit trail.

### Step 5: Evaluate (post-replay)

After replay completes (~40 minutes at 4 events/second), join predictions to
test labels:

```python
import pandas as pd
import sqlite3

# Audit store predictions
with sqlite3.connect("path/to/audit.db") as conn:
    predictions = pd.read_sql(
        "SELECT source_transaction_id, fraud_probability, is_fraud FROM predictions",
        conn
    )

# Test labels
test_labels = pd.read_csv("data/processed/training_features.csv")
test_labels = test_labels.iloc[int(0.85 * len(test_labels)):]

# Join and evaluate
joined = test_labels.merge(
    predictions, 
    left_on="transaction_id",
    right_on="source_transaction_id",
    how="inner"
)

# Metrics
precision = ((joined["is_fraud"] == True) & (joined["is_fraud_pred"] == True)).sum() / (joined["is_fraud_pred"] == True).sum()
recall = ((joined["is_fraud"] == True) & (joined["is_fraud_pred"] == True)).sum() / (joined["is_fraud"] == True).sum()
print(f"Precision: {precision:.3f}, Recall: {recall:.3f}")
```

## Testing

Run all tests (unit + integration):

```bash
.venv/bin/pytest tests/ -v
```

Integration test validates the complete flow:
- Dataset artifacts are consistent
- Training split is chronological
- Model artifact loads and predicts
- Audit records are traceable to source transactions

