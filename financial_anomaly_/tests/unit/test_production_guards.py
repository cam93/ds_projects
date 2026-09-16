import hashlib
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient
from pydantic import ValidationError

from fraud_detector import api
from fraud_detector.features.handbook import FEATURE_NAMES, FeatureState, calculate_features
from fraud_detector.model.inference import FraudScorer
from fraud_detector.schemas import Transaction
from scripts.prepare_dataset import read_source
from scripts.train import average_precision, fit_normalization, split_by_time, train, train_model


def test_missing_model_is_not_ready(monkeypatch):
    monkeypatch.setattr(api.app.state, "scorer", None, raising=False)
    assert TestClient(api.app).get("/ready").status_code == 503


def test_production_rejects_unapproved_model(model_path, monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("MODEL_SHA256", hashlib.sha256(model_path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="release"):
        FraudScorer(model_path)


def test_artifact_integrity_and_normalization(model_path, monkeypatch):
    monkeypatch.setenv("MODEL_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="SHA256"):
        FraudScorer(model_path)
    monkeypatch.delenv("MODEL_SHA256")
    artifact = torch.load(model_path, weights_only=True)
    artifact["normalization_stds"]["amount"] = 0
    torch.save(artifact, model_path)
    with pytest.raises(ValueError, match="normalization"):
        FraudScorer(model_path)


@pytest.mark.parametrize(
    "change",
    [
        {"hour_of_day": 0},
        {"timestamp": "2024-01-01T12:00:00"},
        {"amount": float("inf")},
        {"TX_FRAUD": 1},
        {"customer_id": ""},
    ],
)
def test_schema_rejects_invalid_or_leaking_inputs(transaction, change):
    with pytest.raises(ValidationError):
        Transaction.model_validate({**transaction, **change})


def test_hour_boundary_and_late_event():
    state = FeatureState()
    t = datetime(2024, 1, 1, tzinfo=timezone.utc)

    def calc(stamp):
        return calculate_features(customer_id="c", amount=1, timestamp=stamp, state=state)

    calc(t)
    assert calc(t + timedelta(hours=1))["transactions_last_hour"] == 1
    assert calc(t + timedelta(hours=1, seconds=1))["transactions_last_hour"] == 1
    with pytest.raises(ValueError, match="Out-of-order"):
        calc(t)


def test_pickle_rejected_before_deserialization(tmp_path, monkeypatch):
    path = tmp_path / "untrusted.pkl"
    path.write_bytes(b"not pickle")
    monkeypatch.setattr(pd, "read_pickle", lambda _: pytest.fail("Untrusted pickle deserialized"))
    with pytest.raises(ValueError, match="verified SHA256"):
        read_source(path)


def test_average_precision_ties_are_order_invariant():
    assert average_precision([1, 0], [0.5, 0.5]) == average_precision([0, 1], [0.5, 0.5]) == 0.5


def test_zero_fraud_training_is_rejected():
    data = pd.DataFrame({**{n: [1.0, 2.0] for n in FEATURE_NAMES}, "is_fraud": [0, 0]})
    means, stds = fit_normalization(data)
    with pytest.raises(ValueError, match="both legitimate"):
        train_model(data, data, means, stds, 1)


def test_split_does_not_separate_equal_timestamps():
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=20, tz="UTC").repeat(2),
            "transaction_id": range(40),
        }
    )
    tr, va, te = split_by_time(data)
    assert tr.timestamp.max() < va.timestamp.min() < te.timestamp.min()


def test_release_refuses_insufficient_data(tmp_path):
    data = pd.DataFrame(
        {
            **{n: [1.0] * 20 for n in FEATURE_NAMES},
            "timestamp": pd.date_range("2024-01-01", periods=20, tz="UTC"),
            "transaction_id": range(20),
            "is_fraud": [0] * 20,
        }
    )
    path = tmp_path / "features.csv"
    data.to_csv(path, index=False)
    output = tmp_path / "model.pt"
    with pytest.raises(ValueError, match="Insufficient"):
        train(path, output, policy_path="configs/release-policy.example.json")
    assert not output.exists()


def test_preparation_training_and_release(tmp_path):
    import json

    from scripts.prepare_dataset import prepare_dataset

    source = tmp_path / "source.csv"
    pd.DataFrame(
        {
            "TRANSACTION_ID": [str(i) for i in range(120)],
            "TX_DATETIME": pd.date_range("2024-01-01", periods=120, freq="h", tz="UTC"),
            "CUSTOMER_ID": ["customer"] * 120,
            "TERMINAL_ID": ["terminal"] * 120,
            "TX_AMOUNT": [10.0 if i % 2 == 0 else 1000.0 for i in range(120)],
            "TX_FRAUD": [i % 2 for i in range(120)],
            "TX_FRAUD_SCENARIO": [i % 2 for i in range(120)],
        }
    ).to_csv(source, index=False)
    output = tmp_path / "processed"
    prepare_dataset(source, output)
    policy = tmp_path / "policy.json"
    # Permissive fixture policy tests the mechanism, not real model acceptance.
    policy.write_text(
        json.dumps(
            {
                "min_positive_per_split": 1,
                "min_negative_per_split": 1,
                "min_precision": 0.01,
                "min_recall": 0.01,
                "min_average_precision": 0.01,
            }
        )
    )
    artifact = tmp_path / "candidate.pt"
    report = train(output / "training_features.csv", artifact, epochs=2, policy_path=policy)
    assert report["approved"] is True
    original = artifact.read_bytes()
    with pytest.raises(ValueError, match="immutable"):
        train(output / "training_features.csv", artifact, epochs=2, policy_path=policy)
    assert artifact.read_bytes() == original
    assert artifact.with_suffix(".sha256").exists()
    labels = pd.read_csv(output / "handbook/test_labels.csv")
    assert len(labels) == report["splits"]["test"]["rows"]
    events = [
        json.loads(line) for line in (output / "handbook/replay.jsonl").read_text().splitlines()
    ]
    assert len(events) == len(labels) < 120
    assert all("is_fraud" not in event for event in events)
    assert events[0]["customer_history_days"] > 0


def test_backup_is_readable(tmp_path):
    import sqlite3

    from scripts.maintain_audit import backup_database

    database = tmp_path / "ledger.db"
    backup = tmp_path / "backup.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE sample (value TEXT)")
        connection.execute("INSERT INTO sample VALUES ('durable')")
    backup_database(database, backup)
    with sqlite3.connect(backup) as connection:
        assert connection.execute("SELECT value FROM sample").fetchone()[0] == "durable"
    with pytest.raises(ValueError):
        backup_database(database, backup)
