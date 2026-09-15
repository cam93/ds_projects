# End-to-End Realtime Fraud Detection Flow

## Overview

This document describes the complete journey from raw transaction data to live fraud predictions and audit trails in the realtime fraud detector platform.

## System Architecture

### Five-Container Topology

```
┌────────────────────────────────────────────────────────────────────┐
│                    Production Runtime Stack                         │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐  ┌──────────┐│
│  │   Traffic    │  │     API     │  │    Audit     │  │Prometheus││
│  │  Generator   │→→→   Server    │→→→    Store    │→→→   Server ││
│  │              │  │             │  │              │  │          ││
│  └──────────────┘  └─────────────┘  └──────────────┘  └──────┬───┘│
│                                                                 │    │
│                                                                 v    │
│                                                        ┌──────────────┤
│                                                        │   Grafana    │
│                                                        │  Dashboards  │
│                                                        └──────────────┤
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

## Data Flow Stages

### Stage 1: Data Ingestion and Preparation

**Input**: Raw pickle files containing transactions
- Each transaction: ID, timestamp, amount, customer ID, terminal ID, fraud label

**Process** (`scripts/prepare_dataset.py`):
1. Load pickle file(s) with pandas
2. Validate:
   - All transaction IDs are unique
   - All timestamps are present
   - All amounts are positive
3. Sort deterministically by timestamp, then transaction ID (tie-breaker)
4. Calculate features using `handbook.py`:
   - `amount`: transaction amount
   - `transactions_last_hour`: count of customer's txns in last 60 min
   - `customer_history_days`: days since customer's first transaction
   - `hour_of_day`: hour component of UTC timestamp
5. Chronological train/validation/test split (70/15/15)

**Outputs** → `data/processed/`:
```
records.csv
├─ TRANSACTION_ID, TX_DATETIME, CUSTOMER_ID, TERMINAL_ID, TX_AMOUNT, TX_FRAUD
└─ 9,488 rows (example: 2018-04-01 data)

training_features.csv
├─ transaction_id, customer_id, terminal_id, amount, transactions_last_hour,
│  customer_history_days, hour_of_day, timestamp, is_fraud
└─ Same 9,488 rows with computed features

handbook/replay.jsonl
├─ Test period transactions in JSONL format (one JSON per line)
├─ Ready for HTTP POST to /predict endpoint
└─ ~1,423 rows (15% of 9,488)
```

**Key Property**: Features are calculated **before** adding the current transaction to customer history.
This ensures point-in-time consistency: the features reflect the customer's state *before* this transaction.

### Stage 2: Model Training

**Input**: `training_features.csv` (9,488 rows with features and labels)

**Process** (`scripts/train.py`):

1. **Chronological Split**
   ```
   Training (rows 0-6642):      2018-04-01 00:00 → 14:24
   Validation (rows 6643-7925):  2018-04-01 14:24 → 21:36
   Test (rows 7926-9488):        2018-04-01 21:36 → 23:59
   ```

2. **Normalization** (fit on training period only)
   ```python
   fitted_scaler = StandardScaler().fit(X_train)
   X_train_scaled = fitted_scaler.transform(X_train)
   X_val_scaled = fitted_scaler.transform(X_val)
   X_test_scaled = fitted_scaler.transform(X_test)
   ```

3. **Model Training**
   - Architecture: 2-layer PyTorch MLP (4 inputs → 32 → 1 output)
   - Loss: BCEWithLogitsLoss(pos_weight=negatives/positives)
   - Epochs: 20, optimizer: Adam(lr=1e-3)
   - Early stopping based on validation loss

4. **Threshold Selection**
   - Compute predictions on validation period
   - Select threshold to maximize F1 score
   - If validation has no fraud: default to 0.5

5. **Evaluation on Test Period**
   - Compute precision, recall, average precision
   - Log metrics to console

**Outputs** → `models/artifacts/fraud_model.pt`:
```json
{
  "model_state_dict": {...},      # PyTorch weights
  "scaler_mean": [...],            # Normalization mean
  "scaler_std": [...],             # Normalization std
  "threshold": 0.42,               # Decision threshold
  "model_version": "v1.0",         # Version string
  "feature_names": [...],          # Feature order for validation
  "test_metrics": {                # Final test set metrics
    "precision": 0.667,
    "recall": 0.667,
    "average_precision": 0.750
  }
}
```

**Critical**: The model expects features in a specific order.
The API validates feature order matches `handbook.py` FEATURE_NAMES.

### Stage 3: Runtime Deployment (Docker Compose)

**Container 1: Traffic Generator**
```
Image: realtime-fraud-detector-traffic-generator:dev
Mode: handbook (replay) or random (synthetic)
Input: data/processed/handbook/replay.jsonl (test transactions)
Output: HTTP POST to http://api:8000/predict
Rate: 4 events/second
ID Transformation: 123 → run001:123 (prevents duplicates on re-run)
```

**Container 2: API Server**
```
Image: realtime-fraud-detector-api:dev
Port: 8000
Endpoints:
  - POST /predict (main prediction endpoint)
  - GET /health (liveness check)
  - GET /metrics (Prometheus metrics)

Per Request Flow:
  1. Parse JSON transaction
  2. Validate against Transaction schema
  3. Calculate features using handbook.py
  4. Apply normalization from loaded artifact
  5. Forward to PyTorch model
  6. Apply saved threshold
  7. Emit Prometheus metrics
  8. POST prediction to audit-store:8001/log
  9. Return JSON response
```

**Container 3: Audit Store**
```
Image: realtime-fraud-detector-audit-store:dev
Port: 8001
Database: SQLite at /var/lib/fraud-detector/audit.db
Endpoints:
  - POST /log (log a prediction)
  - GET /predictions (retrieve logged predictions)

Schema:
  - id: integer primary key
  - transaction_id: string (e.g., "run001:123")
  - source_transaction_id: string (e.g., "123") ← for evaluation join
  - timestamp: datetime
  - fraud_probability: float
  - is_fraud: boolean
  - model_version: string
```

**Container 4: Prometheus**
```
Image: prom/prometheus:v2.55.1
Port: 9090
Scrape Target: http://api:8000/metrics
Metrics Collected:
  - fraud_detection:prediction_count (total predictions)
  - fraud_detection:fraud_rate (fraud / total)
  - fraud_detection:prediction_latency_ms (API response time)
  - fraud_detection:model_version (string)
```

**Container 5: Grafana**
```
Image: grafana/grafana:11.3.1
Port: 3000
Datasource: Prometheus at http://prometheus:9090
Dashboards:
  - Fraud Detection Rate (time series)
  - Prediction Latency (histogram)
  - Model Version Tracking (gauge)
  - Predictions Per Second (counter)
```

### Stage 4: Live Prediction and Audit

**Per Transaction (at 4 events/second)**:

```
traffic-generator:                      api:8000:                    audit-store:8001:
  {                                       1. Parse schema                1. Receive prediction
    "transaction_id": "123",              2. Calculate features          2. Store in SQLite
    "customer_id": "C001",           →   3. Normalize                 →  3. Return 200 OK
    "terminal_id": "T001",                4. Model inference
    "amount": 50.00,                      5. Apply threshold
    "timestamp": "2018-04-01T14:30"       6. Emit metrics
  }                                       7. POST audit entry
                                          8. Return prediction
```

**Audit Record**:
```sql
INSERT INTO predictions (
  transaction_id, source_transaction_id, timestamp, 
  fraud_probability, is_fraud, model_version
) VALUES (
  "run001:123",      -- Prefixed to prevent duplicate audit
  "123",             -- Original for evaluation join
  "2018-04-01T14:30:00",
  0.72,              -- Model output
  1,                 -- Thresholded decision
  "v1.0"             -- Model version
);
```

### Stage 5: Evaluation (Post-Replay)

After all 9,488 test transactions are replayed (~40 minutes at 4 events/sec):

**Join Predictions to Test Labels**:
```python
import sqlite3
import pandas as pd

# Read audit trail (predictions from containers)
with sqlite3.connect("audit.db") as conn:
    predictions = pd.read_sql(
        """
        SELECT source_transaction_id, fraud_probability, is_fraud
        FROM predictions
        ORDER BY timestamp
        """,
        conn
    )

# Read test labels (from prepared dataset)
test = pd.read_csv("data/processed/training_features.csv")
test = test.iloc[int(0.85 * len(test)):]  # Last 15% (test period)

# Join on source_transaction_id
joined = test.merge(
    predictions,
    left_on="transaction_id",
    right_on="source_transaction_id",
    how="inner"
)

# Compute metrics
TP = ((joined["is_fraud"] == 1) & (joined["is_fraud"] == 1)).sum()
FP = ((joined["is_fraud"] == 0) & (joined["is_fraud"] == 1)).sum()
FN = ((joined["is_fraud"] == 1) & (joined["is_fraud"] == 0)).sum()

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0

print(f"Precision: {precision:.3f}, Recall: {recall:.3f}")
```

## Feature Consistency

**Critical Invariant**: Features are calculated identically in three contexts:

1. **Dataset Preparation** (`prepare_dataset.py`)
   - Reads `records.csv`
   - Calls `handbook.calculate_features()` for each transaction
   - Outputs `training_features.csv`

2. **Model Training** (`train.py`)
   - Loads `training_features.csv`
   - Validates feature columns match `FEATURE_NAMES`
   - Trains model expecting features in exact order

3. **Live Inference** (`api.py` + `handbook.py`)
   - Receives transaction JSON
   - Calls `handbook.calculate_features()` with current state
   - Validates feature order matches artifact
   - Sends to model in exact same order

**Verification**:
```python
from fraud_detector.features.handbook import FEATURE_NAMES
FEATURE_NAMES == [
    "amount",
    "transactions_last_hour",
    "customer_history_days",
    "hour_of_day"
]
```

If feature order diverges between preparation and inference, model predictions will be incorrect.

## Point-in-Time Correctness

Features are calculated **before** adding the current transaction to customer history:

```python
# Customer state at start: {first_seen: T0, recent_txns: [T-3h, T-1h]}
customer_history_days = now - first_seen  # Days since account opened
transactions_last_hour = len([t for t in recent_txns if t > now - 1h])  # Count before current

# Add current transaction to history
recent_txns.append(now)
```

This prevents **data leakage**: the next transaction's features won't include the current one.

## Deployment Checklist

- [ ] Dataset prepared: `data/processed/records.csv`, `training_features.csv`, `replay.jsonl`
- [ ] Model trained: `models/artifacts/fraud_model.pt` exists
- [ ] Docker images build: `docker compose build`
- [ ] All tests pass: `pytest tests/ -v`
- [ ] Docker Compose stack starts: `docker compose up`
- [ ] API health check passes: `curl http://localhost:8000/health`
- [ ] Prometheus scraping: `curl http://localhost:9090/api/v1/targets`
- [ ] Grafana dashboards visible: `http://localhost:3000`
- [ ] Audit store initialized: `curl http://localhost:8001/predictions`
- [ ] Traffic generator posting: check `docker logs realtime-fraud-detector-traffic-generator`

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| API crashes on first request | Model artifact missing | Run `scripts/train.py` |
| Feature order mismatch error | Handbook.py diverged from model artifact | Verify feature names match |
| No metrics in Prometheus | API not emitting metrics | Check `GET /metrics` returns data |
| Grafana dashboards empty | Datasource not configured | Re-provision datasources in UI |
| Audit store grows slowly | Traffic generator rate too low | Increase TRAFFIC_INTERVAL_SECONDS |
| Docker compose fails to build | .venv in build context | Verify .dockerignore is present |

## Performance Characteristics

**Dataset Preparation** (~1 second):
- Read pickle
- Validate 9,488 transactions
- Calculate features
- Write three artifacts

**Model Training** (~30 seconds):
- Normalize training data
- Train MLP for 20 epochs
- Select threshold on validation
- Export artifact

**Live Prediction**:
- Per-transaction latency: 5-10ms (mean)
- Throughput: 100+ TPS per container
- Memory: ~500MB per API container
- Disk: ~10MB audit DB per 100k predictions

**End-to-End Replay** (~40 minutes):
- 9,488 test transactions at 4 events/second
- Includes network I/O and audit logging
- Deterministic ordering for reproducibility

## Next Steps

1. Run the complete flow locally with Docker Compose
2. Verify Grafana dashboards populate with metrics
3. Inspect audit trail after replay completes
4. Evaluate model performance on held-out test set
5. Deploy to cloud infrastructure using Terraform modules
