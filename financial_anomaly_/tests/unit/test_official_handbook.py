import numpy as np
import pandas as pd

from fraud_detector.features.official_handbook import OFFICIAL_FEATURES, official_features
from scripts.handbook_reference.reference import (
    get_count_risk_rolling_window,
    get_customer_spending_behaviour_features,
    get_train_test_set,
)


def records():
    days = [0, 1, 7, 8, 30, 37, 38]
    return pd.DataFrame(
        {
            "TRANSACTION_ID": range(len(days)),
            "TX_DATETIME": pd.Timestamp("2018-04-01", tz="UTC") + pd.to_timedelta(days, unit="d"),
            "CUSTOMER_ID": [1] * len(days),
            "TERMINAL_ID": [1] * len(days),
            "TX_AMOUNT": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
            "TX_FRAUD": [1, 0, 0, 1, 0, 0, 0],
            "TX_TIME_DAYS": days,
        }
    )


def test_reference_windows_match_including_exact_boundaries():
    raw = records()
    actual = official_features(raw)
    customer = get_customer_spending_behaviour_features(raw.copy())
    terminal = get_count_risk_rolling_window(raw.copy())
    np.testing.assert_allclose(actual[OFFICIAL_FEATURES[3:9]], customer[OFFICIAL_FEATURES[3:9]])
    np.testing.assert_allclose(actual[OFFICIAL_FEATURES[9:]], terminal[OFFICIAL_FEATURES[9:]])
    assert actual.loc[0, "CUSTOMER_ID_NB_TX_30DAY_WINDOW"] == 1
    assert actual.loc[2, "TERMINAL_ID_RISK_1DAY_WINDOW"] == 1
    assert actual.loc[3, "TERMINAL_ID_RISK_1DAY_WINDOW"] == 0


def test_current_and_future_labels_do_not_leak():
    raw = records()
    expected = official_features(raw)
    raw.loc[2:, "TX_FRAUD"] = 1 - raw.loc[2:, "TX_FRAUD"]
    actual = official_features(raw)
    np.testing.assert_allclose(
        expected.loc[:3, OFFICIAL_FEATURES], actual.loc[:3, OFFICIAL_FEATURES]
    )


def test_reference_known_cards_filter_is_separate_from_all_transactions():
    rows = pd.DataFrame(
        {
            "TRANSACTION_ID": range(21),
            "TX_DATETIME": pd.date_range("2018-07-25", periods=21, tz="UTC"),
            "TX_TIME_DAYS": range(21),
            "CUSTOMER_ID": [1] * 21,
            "TX_FRAUD": [1] + [0] * 20,
        }
    )
    train, test = get_train_test_set(rows, pd.Timestamp("2018-07-25", tz="UTC"))
    assert len(train) == 7
    assert len(test) == 0
    assert len(rows[rows.TX_TIME_DAYS >= 14]) == 7
