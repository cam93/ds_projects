"""Shared enrichment and API payload construction for batch and live transactions."""

from datetime import datetime

from fraud_detector.features.handbook import calculate_features
from fraud_detector.schemas import AdaptiveFeatures, BehavioralFeatures, Transaction


def enrich_transaction(raw, state, adaptive):
    timestamp = raw["TX_DATETIME"]
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    # Preserve microsecond precision used by the batch importer.
    if hasattr(timestamp, "to_pydatetime"):
        timestamp = timestamp.to_pydatetime()
    features = calculate_features(
        customer_id=str(raw["CUSTOMER_ID"]),
        terminal_id=str(raw["TERMINAL_ID"]),
        amount=float(raw["TX_AMOUNT"]),
        timestamp=timestamp,
        state=state,
    )
    features.update(adaptive.observe_raw(raw))
    return features


def prepared_transaction(
    *,
    transaction_id,
    customer_id,
    terminal_id,
    timestamp,
    features,
    feedback_delay_days=7,
    source_transaction_id=None,
):
    return Transaction(
        transaction_id=str(transaction_id),
        source_transaction_id=source_transaction_id,
        customer_id=str(customer_id),
        terminal_id=str(terminal_id),
        timestamp=timestamp,
        amount=float(features["amount"]),
        transactions_last_hour=int(features["transactions_last_hour"]),
        customer_history_days=float(features["customer_history_days"]),
        hour_of_day=int(features["hour_of_day"]),
        behavioral_features=BehavioralFeatures.from_features(features),
        adaptive_features=AdaptiveFeatures.from_features(features, feedback_delay_days),
    )
