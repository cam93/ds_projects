"""HTTP integration with a real SQLite ledger and model, without external services."""

import hashlib
import json
import os

import httpx
import pytest
from fastapi.testclient import TestClient

from apps.audit_store import main as audit
from apps.traffic_generator.main import replay_transactions
from fraud_detector import api
from fraud_detector.features.handbook import V2_FEATURE_NAMES, FeatureState, calculate_features
from fraud_detector.model.inference import FraudScorer
from fraud_detector.schemas import BehavioralFeatures, Transaction


@pytest.fixture
def clients(tmp_path, monkeypatch, model_path):
    monkeypatch.setattr(audit, "DATABASE_PATH", str(tmp_path / "audit.db"))
    with TestClient(audit.app) as ledger, TestClient(api.app) as predictor:
        predictor.app.state.audit.close()
        # Authenticated service-to-service calls use the same HTTP interface as production.
        ledger.headers["Authorization"] = "Bearer " + os.environ["AUDIT_WRITE_KEY"]
        predictor.app.state.audit = ledger
        predictor.headers["Authorization"] = "Bearer " + os.environ["API_KEY"]
        yield predictor, ledger


def test_replay_predict_audit_join_and_idempotency(clients, tmp_path, transaction):
    predictor, ledger = clients
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps(transaction) + "\n")
    event = next(replay_transactions(path, "test"))
    first = predictor.post("/predict", json=event)
    assert first.status_code == 200, first.text
    second = predictor.post("/predict", json=event)
    assert second.json() == first.json()
    rows = ledger.get(
        "/predictions", headers={"Authorization": "Bearer " + os.environ["AUDIT_READ_KEY"]}
    ).json()
    assert len(rows) == 1
    labels = {transaction["transaction_id"]: 1}
    assert rows[0]["source_transaction_id"] in labels
    assert rows[0]["fraud_probability"] == first.json()["fraud_probability"]
    assert predictor.get("/ready").status_code == 200
    event["amount"] = 999
    assert predictor.post("/predict", json=event).status_code == 409


def test_authentication_and_roles(clients, transaction):
    predictor, ledger = clients
    assert (
        predictor.post("/predict", json=transaction, headers={"Authorization": ""}).status_code
        == 401
    )
    assert ledger.get("/predictions").status_code == 401  # write key cannot list records
    assert predictor.get("/metrics").status_code == 401
    assert (
        predictor.get(
            "/metrics", headers={"Authorization": "Bearer " + os.environ["METRICS_KEY"]}
        ).status_code
        == 200
    )
    assert predictor.post("/predict", content=b"x" * 17000).status_code == 413


def test_timeout_after_commit_can_be_retried(clients, transaction):
    predictor, ledger = clients

    class LostResponse:
        fail = True

        def request(self, method, path, **kwargs):
            response = ledger.request(method, path, **kwargs)
            if method == "POST" and self.fail:
                self.fail = False
                raise httpx.ReadTimeout("response lost")
            return response

        def close(self):
            pass

    predictor.app.state.audit = LostResponse()
    assert predictor.post("/predict", json=transaction).status_code == 503
    assert predictor.post("/predict", json=transaction).status_code == 200
    rows = ledger.get(
        "/predictions", headers={"Authorization": "Bearer " + os.environ["AUDIT_READ_KEY"]}
    ).json()
    assert len(rows) == 1


def test_ledger_outage_readiness_and_prediction(clients, transaction):
    predictor, _ = clients

    class Unavailable:
        def request(self, *args, **kwargs):
            raise httpx.ConnectError("offline")

        def get(self, *args, **kwargs):
            raise httpx.ConnectError("offline")

        def close(self):
            pass

    predictor.app.state.audit = Unavailable()
    assert predictor.get("/ready").status_code == 503
    assert predictor.post("/predict", json=transaction).status_code == 503


def test_legacy_hash_survives_feature_schema_upgrade(clients, transaction):
    predictor, ledger = clients
    assert predictor.post("/predict", json=transaction).status_code == 200
    saved = ledger.get("/predictions/" + transaction["transaction_id"]).json()
    legacy = Transaction.model_validate(transaction).model_dump(mode="json")
    legacy.pop("behavioral_features")
    expected = hashlib.sha256(
        json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert saved["request_hash"] == expected


def test_v2_model_requires_features_and_persists_prediction(clients, tmp_path, transaction):
    predictor, _ = clients
    path = tmp_path / "v2.json"
    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "feature_names": V2_FEATURE_NAMES,
                "model_type": "logistic",
                "means": [0] * len(V2_FEATURE_NAMES),
                "scales": [1] * len(V2_FEATURE_NAMES),
                "parameters": {"coefficients": [0] * len(V2_FEATURE_NAMES), "intercept": 0},
                "threshold": 0.5,
                "model_version": "v2-http-fixture",
                "release": {"approved": False},
            }
        )
    )
    predictor.app.state.scorer = FraudScorer(path)
    assert predictor.post("/predict", json=transaction).status_code == 422
    event = Transaction.model_validate(transaction)
    features = calculate_features(
        customer_id=event.customer_id,
        terminal_id=event.terminal_id,
        amount=event.amount,
        timestamp=event.timestamp,
        state=FeatureState(),
    )
    transaction["behavioral_features"] = BehavioralFeatures.from_features(features).model_dump()
    response = predictor.post("/predict", json=transaction)
    assert response.status_code == 200, response.text
    assert response.json()["model_version"] == "v2-http-fixture"
    assert predictor.post("/predict", json=transaction).json() == response.json()
    transaction["behavioral_features"]["terminal_mean_amount"] = 5.0
    assert predictor.post("/predict", json=transaction).status_code == 409
