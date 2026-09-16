import pytest
import torch

from fraud_detector.features.handbook import FEATURE_NAMES
from scripts.train import export_artifact


@pytest.fixture(autouse=True)
def service_environment(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL_FILE", raising=False)
    monkeypatch.delenv("MODEL_SHA256", raising=False)
    for index, name in enumerate(("API_KEY", "AUDIT_WRITE_KEY", "AUDIT_READ_KEY", "METRICS_KEY")):
        monkeypatch.delenv(f"{name}_FILE", raising=False)
        monkeypatch.setenv(name, str(index) * 64)


@pytest.fixture
def model_path(tmp_path, monkeypatch):
    path = tmp_path / "model.pt"
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))
    export_artifact(
        model, {n: 0.0 for n in FEATURE_NAMES}, {n: 1.0 for n in FEATURE_NAMES}, 0.5, {}, path
    )
    monkeypatch.setenv("MODEL_PATH", str(path))
    return path


@pytest.fixture
def transaction():
    return {
        "transaction_id": "run:1",
        "source_transaction_id": "1",
        "customer_id": "1",
        "terminal_id": "1",
        "amount": 10.0,
        "transactions_last_hour": 0,
        "customer_history_days": 0.0,
        "hour_of_day": 12,
        "timestamp": "2024-01-01T12:00:00Z",
    }
