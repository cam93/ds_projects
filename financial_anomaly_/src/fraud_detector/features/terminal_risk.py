"""Experimental batch risk features using only prior events and delayed outcomes."""

from fraud_detector.features.adaptive import V3_FEATURE_NAMES

RISK_NAMES = [
    "terminal_volume_1d",
    "terminal_volume_7d",
    "terminal_volume_change",
    "terminal_confirmed_fraud_rate_1d",
    "terminal_confirmed_fraud_rate_change",
    "customer_terminal_familiarity",
]
V4_FEATURE_NAMES = V3_FEATURE_NAMES + RISK_NAMES


def add_terminal_risk(frame, delay_days=7):
    import numpy as np
    import pandas as pd

    if not 1 <= delay_days <= 30:
        raise ValueError("Feedback delay must be in [1,30] days")
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="raise")
    if frame.timestamp.isna().any() or not frame.timestamp.is_monotonic_increasing:
        raise ValueError("Risk features require chronological non-null timestamps")
    if frame.terminal_id.isna().any() or not frame.is_fraud.isin([0, 1]).all():
        raise ValueError("Risk features require terminal IDs and binary outcomes")
    values = np.zeros((len(frame), 5))
    day = 86400 * 10**9
    for positions in frame.groupby("terminal_id", sort=False).indices.values():
        group = frame.iloc[positions]
        stamps = group.timestamp.astype("datetime64[ns, UTC]").astype("int64").to_numpy()
        prior = np.arange(len(group))
        count1 = prior - np.searchsorted(stamps, stamps - day, side="left")
        count7 = prior - np.searchsorted(stamps, stamps - 7 * day, side="left")
        # Inclusive confirmation boundary: original timestamp + delay <= now.
        available = stamps - int(delay_days * day)
        end = np.searchsorted(stamps, available, side="right")
        begin1 = np.searchsorted(stamps, available - day, side="left")
        begin7 = np.searchsorted(stamps, available - 7 * day, side="left")
        positives = np.r_[0, group.is_fraud.to_numpy().cumsum()]
        rate1 = (positives[end] - positives[begin1]) / np.maximum(1, end - begin1)
        rate7 = (positives[end] - positives[begin7]) / np.maximum(1, end - begin7)
        values[positions] = np.column_stack(
            [count1, count7, 7 * count1 / np.maximum(1, count7), rate1, rate1 - rate7]
        )
    frame[RISK_NAMES[:5]] = values
    frame[RISK_NAMES[5]] = frame.customer_terminal_transactions / np.maximum(
        1, frame.customer_prior_transactions
    )
    return frame
