import csv
import json
import sqlite3
from datetime import datetime, timezone

import httpx
import pandas as pd
import pytest

from fraud_detector.schemas import Transaction
from fraud_detector.simulation import COLUMNS, SimulationConfig, TransactionWorld, parse_start
from fraud_detector.simulator import Stream, batch, export_labels, state_lock, stream
from scripts.prepare_dataset import build_replay_events, validate_and_sort


def config(**changes):
    return SimulationConfig(customers=25, terminals=10, **changes)


def responder(request):
    event = json.loads(request.content)
    Transaction.model_validate(event)
    assert "TX_FRAUD" not in event and "is_fraud" not in event
    return httpx.Response(
        200,
        json={
            "transaction_id": event["transaction_id"],
            "source_transaction_id": event["source_transaction_id"],
            "timestamp": event["timestamp"],
            "fraud_probability": 0.25,
            "is_fraud": False,
            "model_version": "test",
        },
    )


def test_seeded_batch_matches_ingestion_and_never_overwrites(tmp_path):
    start = parse_start("2024-01-01")
    cfg = config(
        terminal_compromise_rate=0.2, customer_compromise_rate=0.1, unusual_purchase_fraud_rate=0.05
    )
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    report = batch(cfg, start, 20, first)
    batch(cfg, start, 20, second)
    assert first.read_bytes() == second.read_bytes()
    raw = pd.read_csv(first)
    assert tuple(raw.columns) == COLUMNS
    assert len(raw) == report["rows"] > 1000
    assert 0 < report["fraud"] < report["rows"]
    assert set(raw.TX_FRAUD_SCENARIO) == {0, 1, 2, 3}
    assert raw.TRANSACTION_ID.is_unique
    prepared = validate_and_sort(raw)
    for event in build_replay_events(prepared):
        Transaction.model_validate(event)
    before = first.read_bytes()
    with pytest.raises(ValueError, match="exists"):
        batch(cfg, start, 2, first)
    assert first.read_bytes() == before
    changed = TransactionWorld(config(seed=8), start).day(0)
    assert changed != TransactionWorld(config(), start).day(0)


def test_day_generation_resumes_without_replaying_prior_days():
    world = TransactionWorld(config(), parse_start("2024-01-01"))
    before = world.day(4)
    world.day(0)
    assert world.day(4) == before
    assert TransactionWorld(config(), parse_start("2024-01-01")).day(4) == before
    assert all(row["TX_TIME_DAYS"] == 4 for row in before)


def test_stream_features_match_batch_and_labels_stay_local(tmp_path):
    cfg = config()
    start = parse_start("2024-01-01")
    rows = TransactionWorld(cfg, start).day(0) + TransactionWorld(cfg, start).day(1)
    expected = build_replay_events(validate_and_sort(pd.DataFrame(rows)))
    calls = []

    def handle(request):
        calls.append(json.loads(request.content))
        return responder(request)

    state = tmp_path / "live.db"
    with httpx.Client(base_url="http://test", transport=httpx.MockTransport(handle)) as client:
        stream(cfg, state, "http://test", client, start, count=10, sleep=lambda _: None)
        stream(cfg, state, "http://test", client, start, count=len(rows) - 10, sleep=lambda _: None)
    assert len(calls) == len(rows)
    assert len({event["transaction_id"] for event in calls}) == len(rows)
    for actual, wanted in zip(calls, expected):
        assert actual["source_transaction_id"] == wanted["transaction_id"]
        for name in (
            "amount",
            "transactions_last_hour",
            "customer_history_days",
            "hour_of_day",
            "timestamp",
        ):
            assert actual[name] == wanted[name]
    labels = tmp_path / "labels.csv"
    export_labels(state, labels)
    with labels.open() as source:
        exported = list(csv.DictReader(source))
    assert len(exported) == len(calls)
    assert {row["transaction_id"] for row in exported} == {
        event["transaction_id"] for event in calls
    }


def test_commit_before_checkpoint_failure_retries_identical_request(tmp_path, monkeypatch):
    state = tmp_path / "live.db"
    calls = []

    def handle(request):
        calls.append(json.loads(request.content))
        return responder(request)

    original = Stream.acknowledge

    def lost_ack(*_):
        raise RuntimeError("crash after server commit")

    with httpx.Client(base_url="http://test", transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr(Stream, "acknowledge", lost_ack)
        with pytest.raises(RuntimeError, match="crash"):
            stream(config(), state, "http://test", client, count=1)
        monkeypatch.setattr(Stream, "acknowledge", original)
        stream(config(), state, "http://test", client, count=2, sleep=lambda _: None)
    assert calls[0] == calls[1]
    assert calls[2]["transaction_id"] != calls[1]["transaction_id"]
    with sqlite3.connect(state) as db:
        assert (
            db.execute("SELECT count(*) FROM events WHERE delivered_at IS NOT NULL").fetchone()[0]
            == 2
        )


def test_state_mismatch_and_concurrent_producer_rejected(tmp_path):
    path = tmp_path / "live.db"
    with state_lock(path):
        with pytest.raises(ValueError, match="Another simulator"), state_lock(path):
            pass
        producer = Stream(path, config(), "http://test")
        producer.close()
    with pytest.raises(ValueError, match="different"):
        Stream(path, config(seed=8), "http://test")
    with pytest.raises(ValueError, match="different"):
        Stream(path, config(), "http://other")


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf"), 11])
def test_invalid_rate_rejected_before_creating_state(tmp_path, rate):
    with pytest.raises(ValueError, match="Rate"):
        stream(config(), tmp_path / "live.db", "http://test", None, count=1, rate=rate)
    assert not (tmp_path / "live.db").exists()


def test_configuration_and_timestamp_validation():
    with pytest.raises(ValueError):
        config(transactions_per_customer_day=float("nan"))
    with pytest.raises(ValueError):
        SimulationConfig(customers=0)
    with pytest.raises(ValueError, match="timezone"):
        parse_start("2024-01-01T00:00:00")
    with pytest.raises(ValueError, match="midnight"):
        parse_start("2024-01-01T12:00:00Z")
    assert parse_start("2024-01-01") == datetime(2024, 1, 1, tzinfo=timezone.utc)


def test_zero_fraud_and_controlled_drift():
    cfg = config(
        terminal_compromise_rate=0, customer_compromise_rate=0, unusual_purchase_fraud_rate=0
    )
    start = parse_start("2024-01-01")
    baseline = TransactionWorld(cfg, start)
    drift = TransactionWorld(
        cfg.model_copy(update={"drift_day": 3, "drift_multiplier": 2.0}), start
    )
    assert baseline.day(2) == drift.day(2)
    original, changed = baseline.day(4), drift.day(4)
    assert original and len(original) == len(changed)
    for before, after in zip(original, changed):
        assert before["TX_FRAUD"] == after["TX_FRAUD"] == 0
        assert abs(after["TX_AMOUNT"] - before["TX_AMOUNT"] * 2) <= 0.011


def test_permanent_error_keeps_pending_request_and_no_exported_label(tmp_path):
    path = tmp_path / "live.db"
    with (
        httpx.Client(
            base_url="http://test", transport=httpx.MockTransport(lambda _: httpx.Response(401))
        ) as client,
        pytest.raises(ValueError, match="Permanent"),
    ):
        stream(config(), path, "http://test", client, count=1)
    output = tmp_path / "labels.csv"
    export_labels(path, output)
    assert len(output.read_text().splitlines()) == 1
    with sqlite3.connect(path) as db:
        assert (
            db.execute("SELECT count(*) FROM events WHERE delivered_at IS NULL").fetchone()[0] == 1
        )
