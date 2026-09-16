import json

import httpx
import pytest

from apps.traffic_generator.main import post_transaction, replay_transactions, run_replay


def test_replay_mapping(tmp_path, transaction):
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps(transaction) + "\n")
    event = next(replay_transactions(path, "new"))
    assert event["transaction_id"] == "new:run:1"
    assert event["source_transaction_id"] == "run:1"


def test_permanent_error_is_not_retried(transaction):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(422)

    with (
        httpx.Client(base_url="http://test", transport=httpx.MockTransport(handle)) as client,
        pytest.raises(ValueError, match="Permanent"),
    ):
        post_transaction(client, transaction)
    assert len(calls) == 1


def test_transient_errors_have_bounded_retries(transaction):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(503)

    with (
        httpx.Client(base_url="http://test", transport=httpx.MockTransport(handle)) as client,
        pytest.raises(RuntimeError, match="budget"),
    ):
        post_transaction(client, transaction, attempts=3, sleep=lambda _: None)
    assert len(calls) == 3


def test_checkpoint_resume_and_mismatch(tmp_path, transaction):
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps(transaction) + "\n")
    checkpoint = tmp_path / "checkpoint.json"
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(200, json={})

    with httpx.Client(base_url="http://test", transport=httpx.MockTransport(handle)) as client:
        first = run_replay(path, checkpoint, client, 0)
        second = run_replay(path, checkpoint, client, 0)
        assert first == second and len(calls) == 1
        path.write_text(path.read_text() + "\n")
        with pytest.raises(ValueError, match="Checkpoint"):
            run_replay(path, checkpoint, client, 0)
