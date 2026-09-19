import hashlib
import json
import logging
import os
import ssl
from contextlib import asynccontextmanager
from time import perf_counter
from urllib.parse import quote

import httpx
from fastapi import FastAPI, HTTPException, Request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

from fraud_detector.model.inference import FraudScorer, MissingBehavioralFeatures
from fraud_detector.schemas import AuditRecord, Prediction, Transaction
from fraud_detector.security import ResourceLimits, authorize, secret

logger = logging.getLogger(__name__)
prediction_count = Counter("fraud_predictions_total", "New predictions by decision", ["decision"])
prediction_latency = Histogram(
    "fraud_prediction_latency_seconds", "Prediction request latency including failures"
)
audit_failures = Counter("fraud_audit_failures_total", "Audit failures")
request_count = Counter("fraud_requests_total", "Authenticated requests by outcome", ["outcome"])


@asynccontextmanager
async def lifespan(app):
    app.state.secrets = {name: secret(name) for name in ("API_KEY", "METRICS_KEY")}
    audit_key = secret("AUDIT_WRITE_KEY")
    if len({*app.state.secrets.values(), audit_key}) != 3:
        raise RuntimeError("Use distinct service credentials")
    app.state.scorer = FraudScorer()
    app.state.audit = httpx.Client(
        base_url=os.getenv("AUDIT_SERVICE_URL", "http://audit-store:8001"),
        headers={"Authorization": f"Bearer {audit_key}"},
        timeout=3,
        limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
        trust_env=False,
        verify=ssl.create_default_context(cafile=os.getenv("AUDIT_CA_FILE")),
    )
    yield
    app.state.audit.close()
    app.state.scorer = None


app = FastAPI(
    title="Fraud Detector", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None
)
app.add_middleware(ResourceLimits)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready(request: Request):
    if getattr(request.app.state, "scorer", None) is None:
        raise HTTPException(503, "Model unavailable")
    try:
        response = request.app.state.audit.get("/ready")
        response.raise_for_status()
    except httpx.HTTPError:
        raise HTTPException(503, "Ledger unavailable")
    return {"status": "ready"}


@app.get("/metrics")
def metrics(request: Request):
    authorize(request, "METRICS_KEY")
    return Response(generate_latest(), headers={"Content-Type": CONTENT_TYPE_LATEST})


def ledger_request(client, method, path, **kwargs):
    try:
        response = client.request(method, path, **kwargs)
    except httpx.HTTPError:
        audit_failures.inc()
        raise HTTPException(503, "Ledger unavailable")
    if response.status_code == 409:
        raise HTTPException(409, "Transaction ID conflicts with an existing record")
    if response.status_code not in (200, 201, 404):
        audit_failures.inc()
        raise HTTPException(503, "Ledger unavailable")
    return response


@app.post("/predict", response_model=Prediction)
def predict(transaction: Transaction, request: Request):
    # Features are accepted only from the authenticated, trusted preparation service.
    authorize(request, "API_KEY")
    started = perf_counter()
    outcome = "error"
    try:
        scorer = getattr(request.app.state, "scorer", None)
        if scorer is None:
            raise HTTPException(503, "Model unavailable")
        payload = transaction.idempotency_payload()
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        existing = ledger_request(
            request.app.state.audit,
            "GET",
            f"/predictions/{quote(transaction.transaction_id, safe='')}",
        )
        if existing.status_code == 200:
            record = AuditRecord.model_validate(existing.json())
            if record.request_hash != digest:
                raise HTTPException(409, "Transaction ID already used for different input")
            outcome = "duplicate"
            return Prediction(**record.model_dump(exclude={"request_hash"}))
        try:
            probability, decision = scorer.predict(transaction)
        except MissingBehavioralFeatures as error:
            raise HTTPException(422, str(error)) from None
        record = AuditRecord(
            transaction_id=transaction.transaction_id,
            source_transaction_id=transaction.source_transaction_id,
            timestamp=transaction.timestamp,
            fraud_probability=probability,
            is_fraud=decision,
            model_version=scorer.model_version,
            request_hash=digest,
        )
        result = ledger_request(
            request.app.state.audit, "POST", "/predictions", json=record.model_dump(mode="json")
        )
        saved = AuditRecord.model_validate(result.json())
        if result.status_code == 201:
            prediction_count.labels("fraud" if saved.is_fraud else "legitimate").inc()
        outcome = "success"
        return Prediction(**saved.model_dump(exclude={"request_hash"}))
    finally:
        prediction_latency.observe(perf_counter() - started)
        request_count.labels(outcome).inc()
        logger.info(
            "prediction_request outcome=%s duration_ms=%.1f",
            outcome,
            (perf_counter() - started) * 1000,
        )
