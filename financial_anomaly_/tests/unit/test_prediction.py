from datetime import datetime, timezone

from fraud_detector.model.inference import FraudScorer
from fraud_detector.schemas import Transaction


def test_scorer_returns_probability() -> None:
    transaction = Transaction(
        transaction_id="txn-1",
        customer_id="customer-1",
        terminal_id="terminal-1",
        amount=100.0,
        transactions_last_hour=2,
        customer_history_days=365,
        timestamp=datetime.now(timezone.utc),
    )
    probability = FraudScorer().predict(transaction)
    assert 0 <= probability <= 1
