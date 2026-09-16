"""Point-in-time features; no label, future event or identifier is a model input."""

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

# Preserve v1 artifacts and clients while adding an explicit, complete v2 feature block.
FEATURE_NAMES = ["amount", "transactions_last_hour", "customer_history_days", "hour_of_day"]
BEHAVIORAL_NAMES = [
    "transactions_last_5m",
    "transactions_last_day",
    "customer_prior_transactions",
    "customer_mean_amount",
    "customer_amount_std",
    "amount_to_customer_mean",
    "amount_zscore",
    "is_new_terminal",
    "customer_terminal_transactions",
    "terminal_transactions_last_hour",
    "terminal_mean_amount",
    "amount_to_terminal_mean",
    "hour_sin",
    "hour_cos",
]
V2_FEATURE_NAMES = FEATURE_NAMES + BEHAVIORAL_NAMES
FEATURE_VERSION = 2


@dataclass
class RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def add(self, value):
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    @property
    def std(self):
        return math.sqrt(max(0.0, self.m2 / self.count)) if self.count else 0.0


@dataclass
class FeatureState:
    first_seen: dict[str, datetime] = field(default_factory=dict)
    recent_transactions: dict[str, deque[datetime]] = field(default_factory=dict)
    customer_stats: dict[str, RunningStats] = field(default_factory=dict)
    customer_terminals: dict[str, dict[str, int]] = field(default_factory=dict)
    terminal_stats: dict[str, RunningStats] = field(default_factory=dict)
    terminal_recent: dict[str, deque[datetime]] = field(default_factory=dict)
    latest_timestamp: datetime | None = None


def calculate_features(
    *,
    customer_id: str,
    amount: float,
    timestamp: datetime,
    state: FeatureState,
    terminal_id: str | None = None,
) -> dict[str, float]:
    """Compute against previously processed events, then update history.

    Ties follow the ingestion order (timestamp, lexical transaction ID). Window boundaries
    are inclusive. Customer/terminal aggregates include prior unlabeled observations only.
    Passing no terminal_id retains the original four-field return contract.
    """
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("Timestamp must be timezone aware")
    if not math.isfinite(amount) or not 0 < amount <= 1_000_000:
        raise ValueError("Amount must be finite and within API bounds")
    timestamp = timestamp.astimezone(timezone.utc)
    if state.latest_timestamp is not None and timestamp < state.latest_timestamp:
        raise ValueError("Out-of-order transaction")
    first_seen = state.first_seen.get(customer_id)
    history = state.recent_transactions.setdefault(customer_id, deque())
    day_cutoff = timestamp - timedelta(days=1)
    hour_cutoff = timestamp - timedelta(hours=1)
    while history and history[0] < day_cutoff:
        history.popleft()
    features = {
        "amount": float(amount),
        "transactions_last_hour": float(sum(stamp >= hour_cutoff for stamp in history)),
        "customer_history_days": 0.0
        if first_seen is None
        else (timestamp - first_seen).total_seconds() / 86400,
        "hour_of_day": float(timestamp.hour),
    }
    customer = state.customer_stats.setdefault(customer_id, RunningStats())
    if terminal_id is not None:
        terminal = state.terminal_stats.setdefault(terminal_id, RunningStats())
        terminal_history = state.terminal_recent.setdefault(terminal_id, deque())
        while terminal_history and terminal_history[0] < hour_cutoff:
            terminal_history.popleft()
        visits = state.customer_terminals.setdefault(customer_id, {})
        angle = 2 * math.pi * (timestamp.hour + timestamp.minute / 60) / 24
        features.update(
            {
                "transactions_last_5m": float(
                    sum(stamp >= timestamp - timedelta(minutes=5) for stamp in history)
                ),
                "transactions_last_day": float(len(history)),
                "customer_prior_transactions": float(customer.count),
                "customer_mean_amount": customer.mean,
                "customer_amount_std": customer.std,
                "amount_to_customer_mean": amount / customer.mean if customer.count else 1.0,
                "amount_zscore": max(
                    -1000.0, min(1000.0, (amount - customer.mean) / max(customer.std, 0.01))
                )
                if customer.count >= 2
                else 0.0,
                "is_new_terminal": float(terminal_id not in visits),
                "customer_terminal_transactions": float(visits.get(terminal_id, 0)),
                "terminal_transactions_last_hour": float(len(terminal_history)),
                "terminal_mean_amount": terminal.mean,
                "amount_to_terminal_mean": amount / terminal.mean if terminal.count else 1.0,
                "hour_sin": math.sin(angle),
                "hour_cos": math.cos(angle),
            }
        )
        visits[terminal_id] = visits.get(terminal_id, 0) + 1
        terminal.add(amount)
        terminal_history.append(timestamp)
    customer.add(amount)
    if first_seen is None:
        state.first_seen[customer_id] = timestamp
    history.append(timestamp)
    state.latest_timestamp = timestamp
    return features
