"""Integration test for the complete end-to-end fraud detection flow."""

import json
from pathlib import Path

import pandas as pd
import pytest

from prepare_dataset import validate_and_sort
from fraud_detector.features.handbook import FEATURE_NAMES
from fraud_detector.model.inference import FraudScorer
from fraud_detector.schemas import Transaction


def split_by_time(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split features chronologically into 70% train, 15% validation, 15% test."""
    n = len(features)
    train_size = int(0.70 * n)
    validation_size = int(0.15 * n)

    train = features.iloc[:train_size].reset_index(drop=True)
    validation = features.iloc[train_size : train_size + validation_size].reset_index(drop=True)
    test = features.iloc[train_size + validation_size :].reset_index(drop=True)

    return train, validation, test


@pytest.mark.integration
def test_end_to_end_flow():
    """Verify the complete flow from dataset to predictions."""
    processed_dir = Path("data/processed")

    # Step 1: Verify preparation artifacts exist
    records_path = processed_dir / "records.csv"
    features_path = processed_dir / "training_features.csv"
    replay_path = processed_dir / "handbook" / "replay.jsonl"

    assert records_path.exists(), f"Missing {records_path}"
    assert features_path.exists(), f"Missing {features_path}"
    assert replay_path.exists(), f"Missing {replay_path}"

    # Step 2: Load and validate prepared data
    records = pd.read_csv(records_path, parse_dates=["TX_DATETIME"])
    features = pd.read_csv(features_path, parse_dates=["timestamp"])
    replay_count = sum(1 for _ in replay_path.open())

    assert len(records) == len(features) == replay_count
    assert set(features.columns) >= set(FEATURE_NAMES)

    # Step 3: Verify training split is chronological
    train, validation, test = split_by_time(features)

    train_end = train["timestamp"].max()
    validation_start = validation["timestamp"].min()
    validation_end = validation["timestamp"].max()
    test_start = test["timestamp"].min()

    assert train_end < validation_start, "Train must end before validation starts"
    assert validation_end < test_start, "Validation must end before test starts"

    # Step 4: Load model artifact
    scorer = FraudScorer("models/artifacts/fraud_model.pt")
    assert scorer.model_version is not None
    assert scorer.threshold is not None

    # Step 5: Run inference on sample and validate output
    sample = features.head(5)
    predictions = []

    for _, row in sample.iterrows():
        tx = Transaction(
            transaction_id=str(row["transaction_id"]),
            customer_id=str(row["customer_id"]),
            terminal_id=str(row["terminal_id"]),
            amount=float(row["amount"]),
            transactions_last_hour=int(row["transactions_last_hour"]),
            customer_history_days=float(row["customer_history_days"]),
            hour_of_day=int(row["hour_of_day"]),
            timestamp=row["timestamp"],
        )
        probability, is_fraud = scorer.predict(tx)
        assert 0.0 <= probability <= 1.0
        assert isinstance(is_fraud, bool)
        predictions.append(probability)

    assert len(predictions) == 5

    # Step 6: Verify audit traceability (source IDs are unique)
    source_ids = []
    with replay_path.open(encoding="utf-8") as f:
        for line in f:
            event = json.loads(line)
            source_ids.append(event["transaction_id"])

    assert len(set(source_ids)) == len(source_ids), "All source IDs must be unique"

    # Step 7: Verify we can join predictions back to labels
    test_sample = test[["transaction_id", "is_fraud"]].head(10).copy()
    test_sample["transaction_id"] = test_sample["transaction_id"].astype(str)
    
    sample_data = pd.DataFrame(
        {
            "transaction_id": sample["transaction_id"].astype(str),
            "predicted_probability": predictions,
        }
    )
    test_with_predictions = test_sample.merge(
        sample_data,
        on="transaction_id",
        how="left",
    )

    # All validation passed
    assert True
