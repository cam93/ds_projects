from datetime import datetime, timedelta, timezone

from fraud_detector.features.handbook import FeatureState, calculate_features


def test_features_are_calculated_before_current_transaction_is_recorded() -> None:
    state = FeatureState()
    timestamp = datetime(2024, 1, 1, 10, tzinfo=timezone.utc)

    first = calculate_features(
        customer_id="customer-1",
        amount=10,
        timestamp=timestamp,
        state=state,
    )
    second = calculate_features(
        customer_id="customer-1",
        amount=20,
        timestamp=timestamp + timedelta(minutes=5),
        state=state,
    )

    assert first == {
        "amount": 10.0,
        "transactions_last_hour": 0.0,
        "customer_history_days": 0.0,
        "hour_of_day": 10.0,
    }
    assert second["transactions_last_hour"] == 1.0
    assert second["customer_history_days"] == 5 / 1440
