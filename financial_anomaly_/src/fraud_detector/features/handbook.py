"""Shared, point-in-time transaction feature calculation."""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

FEATURE_NAMES = [
    "amount",
    "transactions_last_hour",
    "customer_history_days",
    "hour_of_day",
]


@dataclass
class FeatureState:
    """Per-customer state retained while transactions are replayed."""

    first_seen: dict[str, datetime] = field(default_factory=dict)
    recent_transactions: dict[str, deque[datetime]] = field(default_factory=dict)


def calculate_features(
    *,
    customer_id: str,
    amount: float,
    timestamp: datetime,
    state: FeatureState,
) -> dict[str, float]:
    """Calculate features before recording the current transaction in history."""
    first_seen = state.first_seen.get(customer_id)
    history = state.recent_transactions.setdefault(customer_id, deque())
    cutoff = timestamp - timedelta(hours=1)
    while history and history[0] < cutoff:
        history.popleft()

    features = {
        "amount": float(amount),
        "transactions_last_hour": float(len(history)),
        "customer_history_days": (
            0.0 if first_seen is None else (timestamp - first_seen).total_seconds() / 86400
        ),
        "hour_of_day": float(timestamp.hour),
    }

    if first_seen is None:
        state.first_seen[customer_id] = timestamp
    history.append(timestamp)
    return features
