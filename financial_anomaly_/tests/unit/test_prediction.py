from datetime import datetime, timezone
from pathlib import Path

import torch

from fraud_detector.model.inference import FraudScorer
from fraud_detector.schemas import Transaction
from scripts.train import export_artifact


def test_scorer_returns_probability(tmp_path: Path) -> None:
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))
    artifact_path = tmp_path / "fraud_model.pt"
    export_artifact(
        model,
        {
            name: 0.0
            for name in ("amount", "transactions_last_hour", "customer_history_days", "hour_of_day")
        },
        {
            name: 1.0
            for name in ("amount", "transactions_last_hour", "customer_history_days", "hour_of_day")
        },
        0.5,
        {},
        artifact_path,
    )
    transaction = Transaction(
        transaction_id="txn-1",
        customer_id="customer-1",
        terminal_id="terminal-1",
        amount=100.0,
        transactions_last_hour=2,
        customer_history_days=365,
        hour_of_day=datetime.now(timezone.utc).hour,
        timestamp=datetime.now(timezone.utc),
    )
    probability, is_fraud = FraudScorer(artifact_path).predict(transaction)
    assert 0 <= probability <= 1
    assert is_fraud == (probability >= 0.5)
