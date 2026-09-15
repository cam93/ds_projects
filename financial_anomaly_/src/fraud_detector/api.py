import json
import os
import urllib.error
import urllib.request
from time import perf_counter

from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, make_asgi_app

from fraud_detector.model.inference import FraudScorer
from fraud_detector.schemas import Prediction, Transaction

app = FastAPI(title="Realtime Fraud Detector", version="0.1.0")
scorer = None
audit_service_url = os.getenv("AUDIT_SERVICE_URL", "http://localhost:8001")
prediction_count = Counter("fraud_predictions_total", "Predictions by decision", ["decision"])
prediction_latency = Histogram("fraud_prediction_latency_seconds", "Prediction latency")
audit_failures = Counter("fraud_audit_failures_total", "Audit write failures")
app.mount("/metrics", make_asgi_app())


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    if scorer is None and not os.path.exists(
        os.getenv("MODEL_PATH", "models/artifacts/fraud_model.pt")
    ):
        return {"status": "not_ready"}
    return {"status": "ready"}


@app.post("/predict", response_model=Prediction)
def predict(transaction: Transaction) -> Prediction:
    started = perf_counter()
    global scorer
    if scorer is None:
        scorer = FraudScorer()
    probability, is_fraud = scorer.predict(transaction)
    prediction = Prediction(
        transaction_id=transaction.transaction_id,
        source_transaction_id=transaction.source_transaction_id,
        timestamp=transaction.timestamp,
        fraud_probability=probability,
        is_fraud=is_fraud,
        model_version=scorer.model_version,
    )
    request = urllib.request.Request(
        f"{audit_service_url.rstrip('/')}/predictions",
        data=json.dumps(prediction.model_dump(mode="json")).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=2):
            pass
    except (urllib.error.URLError, TimeoutError) as error:
        audit_failures.inc()
        raise HTTPException(status_code=503, detail="audit service unavailable") from error
    prediction_count.labels("fraud" if prediction.is_fraud else "legitimate").inc()
    prediction_latency.observe(perf_counter() - started)
    return prediction
