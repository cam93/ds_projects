import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pandas as pd
import pytest
from prometheus_client import CollectorRegistry, generate_latest

from fraud_detector.features.adaptive import V3_FEATURE_NAMES, AdaptiveState
from fraud_detector.model.inference import FraudScorer, MissingBehavioralFeatures
from fraud_detector.quality import QualityCollector, initialize_quality, start_metrics
from fraud_detector.schemas import AdaptiveFeatures, BehavioralFeatures, Transaction
from fraud_detector.simulation import SimulationConfig, parse_start
from fraud_detector.simulator import Stream
from scripts.prepare_dataset import build_training_features, validate_and_sort
from scripts.robust_evaluation import select, split_window, validate_policy, wilson


def test_feedback_is_released_only_at_delay_and_current_outcome_excluded():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    state = AdaptiveState(7)

    def event(days, outcome=0, amount=10):
        return state.observe(
            customer_id="c",
            terminal_id="t",
            amount=amount,
            timestamp=start + timedelta(days=days),
            outcome=outcome,
        )

    assert event(0, 1)["terminal_confirmed_fraud_rate"] == 0
    assert event(6.999, 1)["terminal_feedback_count"] == 0
    released = event(7, 0, 999)
    assert released["terminal_confirmed_fraud_rate"] == 1
    assert released["terminal_feedback_count"] == 1
    assert released["customer_mean_amount_1d"] == 10
    assert released["customer_mean_amount_7d"] == 10  # current amount excluded
    restored = AdaptiveState(7, state.snapshot())
    assert restored.snapshot() == state.snapshot()
    # Outcomes received more than seven days ago expire, including when an entity is idle.
    assert event(30)["terminal_feedback_count"] == 0
    with pytest.raises(ValueError, match="Out-of-order"):
        event(29)


def test_stream_v3_matches_offline_across_feedback_boundary_and_restart(tmp_path):
    path = tmp_path / "stream.db"
    config = SimulationConfig(customers=2, terminals=2, terminal_compromise_rate=0.5)
    captured = []
    for count in (55, 55):
        stream = Stream(path, config, "http://test", parse_start("2024-01-01"))
        for _ in range(count):
            request = stream.pending()
            captured.append(request)
            response = {
                "transaction_id": request["transaction_id"],
                "timestamp": request["timestamp"],
                "is_fraud": False,
                "fraud_probability": 0.1,
                "model_version": "v3-test",
            }
            stream.acknowledge(request, response)
            stream.acknowledge(request, response)  # retry must not double-count quality
        stream.close()
    import sqlite3

    with sqlite3.connect(path) as db:
        raw = pd.DataFrame(
            [json.loads(row[0]) for row in db.execute("SELECT raw_json FROM events ORDER BY rowid")]
        )
        assert db.execute("SELECT sum(count) FROM quality").fetchone()[0] == 110
        # Reconstruct quality aggregates for an older journal exactly once.
        db.execute("DROP TABLE quality")
        initialize_quality(db)
        assert db.execute("SELECT sum(count) FROM quality").fetchone()[0] == 110
    expected = build_training_features(validate_and_sort(raw))
    assert (
        pd.to_datetime(raw.TX_DATETIME, utc=True).max()
        - pd.to_datetime(raw.TX_DATETIME, utc=True).min()
    ).days >= 7
    for index, event in enumerate(captured):
        assert (
            event["adaptive_features"]
            == AdaptiveFeatures.from_features(expected.iloc[index]).model_dump()
        )
    registry = CollectorRegistry()
    registry.register(QualityCollector(path))
    content = generate_latest(registry).decode()
    assert 'model_version="v3-test"' in content and 'outcome="fp"' in content
    assert "transaction_id" not in content


def test_quality_endpoint_auth_and_unavailable_journal(tmp_path):
    server = start_metrics(tmp_path / "absent.db", 0, "k" * 32, host="127.0.0.1")
    try:
        with httpx.Client(
            base_url=f"http://127.0.0.1:{server.server_port}", trust_env=False
        ) as client:
            assert client.get("/metrics").status_code == 401
            assert (
                client.get("/metrics", headers={"Authorization": "Bearer " + "k" * 32}).status_code
                == 503
            )
            assert client.get("/elsewhere").status_code == 404
    finally:
        server.shutdown()
        server.server_close()


def test_v3_requires_matching_delay_and_complete_fields(tmp_path, transaction):
    artifact = {
        "format_version": 1,
        "feature_names": V3_FEATURE_NAMES,
        "model_type": "logistic",
        "means": [0] * len(V3_FEATURE_NAMES),
        "scales": [1] * len(V3_FEATURE_NAMES),
        "parameters": {"coefficients": [0] * len(V3_FEATURE_NAMES), "intercept": 0},
        "threshold": 0.5,
        "feedback_delay_days": 7,
        "model_version": "v3-fixture",
        "release": {"approved": False},
    }
    path = tmp_path / "model.json"
    path.write_text(json.dumps(artifact))
    scorer = FraudScorer(path)
    with pytest.raises(MissingBehavioralFeatures):
        scorer.predict(Transaction.model_validate(transaction))
    from fraud_detector.features.handbook import FeatureState, calculate_features

    event = Transaction.model_validate(transaction)
    b = calculate_features(
        customer_id=event.customer_id,
        terminal_id=event.terminal_id,
        amount=event.amount,
        timestamp=event.timestamp,
        state=FeatureState(),
    )
    a = AdaptiveState().observe(
        customer_id=event.customer_id,
        terminal_id=event.terminal_id,
        amount=event.amount,
        timestamp=event.timestamp,
        outcome=1,
    )
    transaction["behavioral_features"] = BehavioralFeatures.from_features(b).model_dump()
    transaction["adaptive_features"] = AdaptiveFeatures.from_features(a).model_dump()
    assert scorer.predict(Transaction.model_validate(transaction)) == (0.5, True)
    transaction["adaptive_features"]["feedback_delay_days"] = 1
    with pytest.raises(MissingBehavioralFeatures):
        scorer.predict(Transaction.model_validate(transaction))


def test_rolling_selection_uses_worst_interval_and_delayed_label_cutoffs():
    def fold(recall, upper):
        return {
            "metrics": {
                "recall": recall,
                "precision": 0.5,
                "false_positive_rate": upper / 2,
                "fpr_95_interval": [0, upper],
            }
        }

    winner, _, ok = select(
        {"high_recall": [fold(0.9, 0.012)], "stable": [fold(0.5, 0.008), fold(0.6, 0.009)]}, 0.01
    )
    assert winner == "stable" and ok
    winner, _, ok = select({"a": [fold(0.9, 0.012)], "b": [fold(0.5, 0.011)]}, 0.01)
    assert winner == "b" and not ok
    low, upper = wilson(0, 100)
    assert low == 0 and 0 < upper < 0.05
    assert wilson(0, 0) == [None, None]
    p = json.loads(Path("configs/robust-evaluation.json").read_text())
    validate_policy(p)
    frame = pd.DataFrame({"day": range(112), "is_fraud": [i % 2 for i in range(112)]})
    train, calibration, evaluation = split_window(frame, 42, p)
    assert train.day.max() == 34 and calibration.day.min() == 42
    assert calibration.day.max() == 55 and evaluation.day.min() == 63
    p["final_seed"] = p["training_seeds"][0]
    with pytest.raises(ValueError):
        validate_policy(p)
