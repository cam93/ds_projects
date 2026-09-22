import numpy as np
import pandas as pd
import pytest

from fraud_detector.features.terminal_risk import RISK_NAMES, add_terminal_risk
from scripts.tune_rolling import choose, operating_threshold, windows


def sample():
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01", "2024-01-07", "2024-01-08", "2024-01-09", "2024-01-20"], utc=True
            ),
            "terminal_id": ["t"] * 5,
            "is_fraud": [1, 0, 0, 1, 1],
            "customer_terminal_transactions": [0, 1, 2, 3, 4],
            "customer_prior_transactions": [0, 2, 4, 6, 8],
        }
    )


def test_delayed_outcomes_and_volume_boundaries():
    actual = add_terminal_risk(sample())
    assert actual.terminal_confirmed_fraud_rate_1d.tolist() == [0, 0, 1, 1, 0]
    assert actual.terminal_volume_1d.tolist() == [0, 0, 1, 1, 0]
    assert actual.terminal_volume_7d.tolist() == [0, 1, 2, 2, 0]
    assert actual.customer_terminal_familiarity.tolist() == [0, 0.5, 0.5, 0.5, 0.5]


def test_future_rows_and_unavailable_labels_cannot_change_features():
    original = sample()
    before = add_terminal_risk(original)
    altered = original.copy()
    altered.loc[1:, "is_fraud"] = 1 - altered.loc[1:, "is_fraud"]
    after = add_terminal_risk(altered)
    pd.testing.assert_frame_equal(before.loc[:3, RISK_NAMES], after.loc[:3, RISK_NAMES])
    pd.testing.assert_frame_equal(
        before.loc[:2, RISK_NAMES], add_terminal_risk(original.iloc[:3])[RISK_NAMES]
    )


def test_terminal_history_is_isolated_and_timestamp_order_required():
    frame = sample()
    frame.loc[2, "terminal_id"] = "other"
    result = add_terminal_risk(frame)
    assert result.loc[2, "terminal_volume_7d"] == 0
    assert result.loc[2, "terminal_confirmed_fraud_rate_1d"] == 0
    with pytest.raises(ValueError):
        add_terminal_risk(frame.iloc[::-1])


def test_rolling_windows_have_label_gaps_and_warmup():
    frame = pd.DataFrame({"day": np.arange(120)})
    train, calibration, evaluation = windows(frame, 84)
    assert (train.day.min(), train.day.max()) == (14, 55)
    assert (calibration.day.min(), calibration.day.max()) == (63, 76)
    assert (evaluation.day.min(), evaluation.day.max()) == (84, 97)


def test_selection_requires_every_fold_above_target():
    results = {
        "good": {
            "summary": {"minimum_recall": 0.86, "mean_precision": 0.3, "worst_alerts_per_1000": 20}
        },
        "bad": {
            "summary": {"minimum_recall": 0.85, "mean_precision": 0.99, "worst_alerts_per_1000": 1}
        },
    }
    assert choose(results) == "good"
    assert choose({"bad": results["bad"]}) is None


def test_threshold_uses_precision_and_preserves_ties():
    assert operating_threshold([1, 1, 1, 0], [0.9, 0.8, 0.8, 0.8], 0.9) == 0.8


def test_timestamp_resolution_does_not_change_windows():
    frame = sample()
    expected = add_terminal_risk(frame)
    frame["timestamp"] = frame.timestamp.astype("datetime64[us, UTC]")
    pd.testing.assert_frame_equal(expected[RISK_NAMES], add_terminal_risk(frame)[RISK_NAMES])


def test_experimental_schema_portable_parity_and_live_rejection(tmp_path, monkeypatch):
    import json

    from sklearn.ensemble import HistGradientBoostingClassifier

    from fraud_detector.features.terminal_risk import V4_FEATURE_NAMES
    from fraud_detector.model.candidates import FittedCandidate
    from fraud_detector.model.inference import FraudScorer
    from scripts.tune_rolling import artifact

    rng = np.random.default_rng(7)
    frame = pd.DataFrame(rng.normal(size=(80, len(V4_FEATURE_NAMES))), columns=V4_FEATURE_NAMES)
    labels = (frame.iloc[:, -1] > 0).astype(int)
    model = HistGradientBoostingClassifier(max_iter=3, min_samples_leaf=3).fit(
        frame.to_numpy(), labels
    )
    fitted = FittedCandidate(
        model,
        V4_FEATURE_NAMES,
        "hist_gradient_boosting",
        dict.fromkeys(V4_FEATURE_NAMES, 0.0),
        dict.fromkeys(V4_FEATURE_NAMES, 1.0),
        {},
    )
    exported = artifact(fitted, 0.5, "experiment")
    assert fitted.verify_export(frame, fitted.score(frame), exported) < 1e-6
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(exported))
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.delenv("MODEL_SHA256", raising=False)
    with pytest.raises(ValueError, match="feature order"):
        FraudScorer(path)
