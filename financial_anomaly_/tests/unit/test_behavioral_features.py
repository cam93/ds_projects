import json
import sqlite3
from datetime import datetime, timedelta, timezone

import httpx
import pandas as pd
import pytest

from fraud_detector.features.handbook import V2_FEATURE_NAMES, FeatureState, calculate_features
from fraud_detector.schemas import BehavioralFeatures, Transaction
from fraud_detector.simulation import SimulationConfig, parse_start
from fraud_detector.simulator import Stream, stream
from scripts.prepare_dataset import build_training_features, validate_and_sort


def test_windows_stats_and_novelty_use_only_prior_observations():
    state = FeatureState()
    t = datetime(2024, 1, 1, tzinfo=timezone.utc)

    def event(stamp, amount, terminal="t"):
        return calculate_features(
            customer_id="c", terminal_id=terminal, amount=amount, timestamp=stamp, state=state
        )

    first = event(t, 10)
    assert set(first) == set(V2_FEATURE_NAMES)
    assert first["customer_mean_amount"] == first["customer_prior_transactions"] == 0
    assert first["amount_to_customer_mean"] == 1
    assert first["is_new_terminal"] == 1
    second = event(t + timedelta(minutes=5), 30)
    assert second["transactions_last_5m"] == second["transactions_last_hour"] == 1
    assert second["customer_mean_amount"] == 10  # Excludes the current amount of 30.
    assert second["customer_amount_std"] == 0
    assert second["amount_to_customer_mean"] == 3
    assert second["customer_terminal_transactions"] == 1 and second["is_new_terminal"] == 0
    third = event(t + timedelta(hours=1), 40, "new")
    assert third["transactions_last_hour"] == 2
    assert third["transactions_last_5m"] == 0
    assert third["customer_mean_amount"] == 20 and third["customer_amount_std"] == 10
    assert third["amount_zscore"] == 2
    assert third["terminal_mean_amount"] == 0 and third["is_new_terminal"] == 1
    fourth = event(t + timedelta(days=1, seconds=1), 50)
    assert fourth["transactions_last_day"] == 2
    assert fourth["transactions_last_hour"] == 0
    assert fourth["customer_prior_transactions"] == 3


def test_future_events_and_labels_do_not_change_past_features():
    raw = pd.DataFrame(
        [
            {
                "TRANSACTION_ID": str(i),
                "TX_DATETIME": f"2024-01-01T0{i}:00:00Z",
                "CUSTOMER_ID": "c",
                "TERMINAL_ID": "t",
                "TX_AMOUNT": float(i + 10),
                "TX_FRAUD": i % 2,
                "TX_FRAUD_SCENARIO": i % 2,
            }
            for i in range(4)
        ]
    )
    before = build_training_features(validate_and_sort(raw))
    changed = raw.copy()
    changed["TX_FRAUD"] = 1 - changed.TX_FRAUD
    changed["TX_FRAUD_SCENARIO"] = changed.TX_FRAUD * 3
    changed.loc[3, "TX_AMOUNT"] = 999999
    after = build_training_features(validate_and_sort(changed))
    pd.testing.assert_frame_equal(
        before.iloc[:3][V2_FEATURE_NAMES], after.iloc[:3][V2_FEATURE_NAMES]
    )
    prior_columns = [
        n
        for n in V2_FEATURE_NAMES
        if n
        not in ("amount", "amount_to_customer_mean", "amount_zscore", "amount_to_terminal_mean")
    ]
    pd.testing.assert_series_equal(before.iloc[3][prior_columns], after.iloc[3][prior_columns])


def test_migrate_v1_stream_history_without_changing_pending_request(tmp_path):
    path = tmp_path / "live.db"
    config = SimulationConfig(customers=10, terminals=3)
    producer = Stream(path, config, "http://test", parse_start("2024-01-01"))
    pending = producer.pending()
    pending.pop("behavioral_features")
    with producer.db:
        producer.state.pop("feature_version")
        producer.state["features"] = {"first": {}, "recent": {}}
        producer._save()
        producer.db.execute("UPDATE events SET request_json=?", (json.dumps(pending),))
    producer.close()
    restored = Stream(path, config, "http://test")
    try:
        assert restored.pending() == pending
        assert sum(value.count for value in restored.features.customer_stats.values()) == 1
        assert restored.state["feature_version"] == 2
    finally:
        restored.close()


def test_stream_extended_features_survive_restart_and_match_offline(tmp_path):
    config = SimulationConfig(customers=10, terminals=3)
    path = tmp_path / "live.db"
    captured = []

    def handle(request):
        event = json.loads(request.content)
        captured.append(event)
        return httpx.Response(
            200,
            json={
                "transaction_id": event["transaction_id"],
                "timestamp": event["timestamp"],
                "source_transaction_id": event["source_transaction_id"],
                "fraud_probability": 0.2,
                "is_fraud": False,
                "model_version": "test",
            },
        )

    with httpx.Client(base_url="http://test", transport=httpx.MockTransport(handle)) as client:
        stream(
            config,
            path,
            "http://test",
            client,
            parse_start("2024-01-01"),
            count=70,
            sleep=lambda _: None,
        )
        stream(config, path, "http://test", client, count=70, sleep=lambda _: None)
    with sqlite3.connect(path) as db:
        rows = [
            json.loads(row[0]) for row in db.execute("SELECT raw_json FROM events ORDER BY rowid")
        ]
    offline = build_training_features(validate_and_sort(pd.DataFrame(rows)))
    for event, (_, row) in zip(captured, offline.iterrows()):
        expected = BehavioralFeatures.from_features(row).model_dump()
        assert event["behavioral_features"] == expected
        Transaction.model_validate(event)


def test_partial_behavioral_block_is_rejected(transaction):
    with pytest.raises(ValueError):
        Transaction.model_validate({**transaction, "behavioral_features": {"is_new_terminal": 1}})
