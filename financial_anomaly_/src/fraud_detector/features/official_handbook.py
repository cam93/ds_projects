"""Batch implementation of the Handbook's 15 reference inputs.

Customer windows include the current transaction; delayed terminal label windows
are (t-delay-window, t-delay]. These are distinct from the live model's schema.
"""

import numpy as np
import pandas as pd

OFFICIAL_FEATURES = (
    ["TX_AMOUNT", "TX_DURING_WEEKEND", "TX_DURING_NIGHT"]
    + [
        f"CUSTOMER_ID_{kind}_{days}DAY_WINDOW"
        for days in (1, 7, 30)
        for kind in ("NB_TX", "AVG_AMOUNT")
    ]
    + [f"TERMINAL_ID_{kind}_{days}DAY_WINDOW" for days in (1, 7, 30) for kind in ("NB_TX", "RISK")]
)


def official_features(records):
    frame = records.sort_values("TRANSACTION_ID").reset_index(drop=True).copy()
    frame["TX_DATETIME"] = pd.to_datetime(frame.TX_DATETIME, utc=True)
    if not frame.TX_DATETIME.is_monotonic_increasing or frame.TRANSACTION_ID.duplicated().any():
        raise ValueError("Expected unique chronological transaction IDs")
    if not frame.TX_FRAUD.isin([0, 1]).all() or not np.isfinite(frame.TX_AMOUNT).all():
        raise ValueError("Invalid labels or amounts")
    frame["TX_DURING_WEEKEND"] = (frame.TX_DATETIME.dt.weekday >= 5).astype(int)
    frame["TX_DURING_NIGHT"] = (frame.TX_DATETIME.dt.hour <= 6).astype(int)
    day = 86400 * 10**9
    customer = np.zeros((len(frame), 6))
    terminal = np.zeros((len(frame), 6))
    for positions in frame.groupby("CUSTOMER_ID", sort=False).indices.values():
        g = frame.iloc[positions]
        stamps = g.TX_DATETIME.astype("datetime64[ns, UTC]").astype("int64").to_numpy()
        end = np.arange(len(g)) + 1
        amounts = np.r_[0, g.TX_AMOUNT.to_numpy().cumsum()]
        for column, window in enumerate((1, 7, 30)):
            begin = np.searchsorted(stamps, stamps - window * day, side="right")
            count = end - begin
            customer[positions, column * 2] = count
            customer[positions, column * 2 + 1] = (amounts[end] - amounts[begin]) / count
    for positions in frame.groupby("TERMINAL_ID", sort=False).indices.values():
        g = frame.iloc[positions]
        stamps = g.TX_DATETIME.astype("datetime64[ns, UTC]").astype("int64").to_numpy()
        labels = np.r_[0, g.TX_FRAUD.to_numpy().cumsum()]
        end = np.searchsorted(stamps, stamps - 7 * day, side="right")
        for column, window in enumerate((1, 7, 30)):
            begin = np.searchsorted(stamps, stamps - (7 + window) * day, side="right")
            count = end - begin
            terminal[positions, column * 2] = count
            terminal[positions, column * 2 + 1] = (labels[end] - labels[begin]) / np.maximum(
                1, count
            )
    frame[OFFICIAL_FEATURES[3:9]] = customer
    frame[OFFICIAL_FEATURES[9:]] = terminal
    return frame
