import hashlib
import json
from pathlib import Path

from fraud_detector.features.handbook import V2_FEATURE_NAMES
from fraud_detector.model.inference import FraudScorer


def test_default_demo_uses_pinned_gradient_boosting(monkeypatch):
    monkeypatch.delenv("MODEL_PATH", raising=False)
    path = Path("models/artifacts/fraud_model.json")
    artifact = json.loads(path.read_text())
    assert artifact["model_type"] == "hist_gradient_boosting"
    assert artifact["feature_names"] == V2_FEATURE_NAMES
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == path.with_suffix(".sha256").read_text().strip()
    )
    assert not Path("models/artifacts/fraud_model.pt").exists()
    assert FraudScorer().model_version == artifact["model_version"]
    assert (
        artifact["release"]["approved"] is False
    )  # Demo promotion does not bypass production gates.
