"""Rolling spending and delayed investigation feedback for synthetic demo evaluation."""

from collections import deque
from datetime import datetime, timezone

from fraud_detector.features.handbook import V2_FEATURE_NAMES

ADAPTIVE_NAMES = [
    "customer_mean_amount_1d",
    "customer_mean_amount_7d",
    "customer_spending_change",
    "terminal_mean_amount_1d",
    "terminal_mean_amount_7d",
    "terminal_spending_change",
    "customer_confirmed_fraud_rate",
    "terminal_confirmed_fraud_rate",
    "customer_feedback_count",
    "terminal_feedback_count",
]
V3_FEATURE_NAMES = V2_FEATURE_NAMES + ADAPTIVE_NAMES


class AdaptiveState:
    """Only release prior outcomes at event time + delay; never use today's outcome today.

    Spending windows cover preceding 1/7 days. Feedback rates cover confirmations received
    in the preceding 7 days, including legitimate outcomes in their denominators.
    """

    def __init__(self, delay_days=7, saved=None):
        if not isinstance(delay_days, (float, int)) or not 1 <= delay_days <= 30:
            raise ValueError("Feedback delay must be between 1 and 30 days")
        self.delay_days = delay_days
        saved = saved or {}
        self.latest = saved.get("latest")
        self.pending = deque(saved.get("pending", []))
        self.spending = {
            kind: {
                key: deque(rows) for key, rows in saved.get("spending", {}).get(kind, {}).items()
            }
            for kind in ("customer", "terminal")
        }
        self.feedback = {
            kind: {
                key: deque(rows) for key, rows in saved.get("feedback", {}).get(kind, {}).items()
            }
            for kind in ("customer", "terminal")
        }

    def snapshot(self):
        return {
            "latest": self.latest,
            "delay_days": self.delay_days,
            "pending": list(self.pending),
            **{
                name: {
                    kind: {key: list(rows) for key, rows in groups.items()}
                    for kind, groups in getattr(self, name).items()
                }
                for name in ("spending", "feedback")
            },
        }

    def observe(self, *, customer_id, terminal_id, amount, timestamp, outcome):
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("Feedback requires timezone-aware timestamps")
        t = timestamp.timestamp()
        if self.latest is not None and t < self.latest:
            raise ValueError("Out-of-order feedback event")
        if outcome not in (0, 1):
            raise ValueError("Investigation outcome must be binary")
        while self.pending and self.pending[0][0] <= t:
            available, customer, terminal, label = self.pending.popleft()
            if available < t - 7 * 86400:
                continue
            for kind, key in (("customer", customer), ("terminal", terminal)):
                self.feedback[kind].setdefault(key, deque()).append((available, label))
        result = {}
        for kind, key in (("customer", str(customer_id)), ("terminal", str(terminal_id))):
            history = self.spending[kind].setdefault(key, deque())
            while history and history[0][0] < t - 7 * 86400:
                history.popleft()
            recent = [value for stamp, value in history if stamp >= t - 86400]
            week = sum(value for _, value in history) / len(history) if history else 0.0
            day = sum(recent) / len(recent) if recent else 0.0
            result[f"{kind}_mean_amount_1d"] = day
            result[f"{kind}_mean_amount_7d"] = week
            result[f"{kind}_spending_change"] = day / week if recent and week else 1.0
            known = self.feedback[kind].setdefault(key, deque())
            while known and known[0][0] < t - 7 * 86400:
                known.popleft()
            result[f"{kind}_feedback_count"] = len(known)
            result[f"{kind}_confirmed_fraud_rate"] = (
                sum(label for _, label in known) / len(known) if known else 0.0
            )
            history.append((t, amount))
        # Deliberately enqueue AFTER computing features. This value stays in the local oracle.
        self.pending.append(
            (t + self.delay_days * 86400, str(customer_id), str(terminal_id), int(outcome))
        )
        self.latest = t
        return result

    def observe_raw(self, raw):
        stamp = raw["TX_DATETIME"]
        if isinstance(stamp, str):
            stamp = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        return self.observe(
            customer_id=str(raw["CUSTOMER_ID"]),
            terminal_id=str(raw["TERMINAL_ID"]),
            amount=float(raw["TX_AMOUNT"]),
            timestamp=stamp.astimezone(timezone.utc),
            outcome=int(raw["TX_FRAUD"]),
        )
