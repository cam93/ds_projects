from datetime import datetime, timezone

from fraud_detector.model.inference import FraudScorer
from fraud_detector.schemas import Transaction


def test_scorer_returns_probability() -> None:
    transaction = Transaction(
        transaction_id="txn-1",
        amount=100.0,
        account_age_days=365,
        transactions_last_hour=2,
        is_international=False,
        timestamp=datetime.now(timezone.utc),
    )
    probability = FraudScorer().predict(transaction)
    assert 0 <= probability <= 1
