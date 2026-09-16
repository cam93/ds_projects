import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from fraud_detector.features.handbook import FEATURE_NAMES
from fraud_detector.model.inference import FraudScorer, MissingBehavioralFeatures
from fraud_detector.model.portable import PortableModel
from fraud_detector.schemas import Transaction
from fraud_detector.simulation import SimulationConfig, TransactionWorld, parse_start
from scripts.compare_models import choose, export_parameters, run, threshold_at_fpr
from scripts.prepare_dataset import build_training_features, validate_and_sort


def test_threshold_respects_cap_without_splitting_score_ties():
    assert threshold_at_fpr([1, 0, 1, 0], [0.9, 0.9, 0.8, 0.1], 0.0) == 2.0
    threshold = threshold_at_fpr([1, 0, 1, 0], [0.9, 0.9, 0.8, 0.1], 0.5)
    assert threshold == 0.8
    assert threshold_at_fpr([1, 1, 0, 0], [0.9, 0.8, 0.7, 0.6], 0.0) == 0.8


def test_selection_is_validation_only():
    assert (
        choose(
            {
                "a": {"recall": 0.4, "precision": 0.5, "average_precision": 0.7},
                "b": {"recall": 0.5, "precision": 0.3, "average_precision": 0.4},
            }
        )
        == "b"
    )


@pytest.mark.parametrize("kind", ["logistic", "hist_gradient_boosting"])
def test_json_inference_matches_native_models(kind, tmp_path, transaction):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(100, 4))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    model = (
        LogisticRegression().fit(x, y)
        if kind == "logistic"
        else HistGradientBoostingClassifier(
            max_iter=5, min_samples_leaf=5, early_stopping=False
        ).fit(x, y)
    )
    artifact = {
        "format_version": 1,
        "model_type": kind,
        "feature_names": FEATURE_NAMES,
        "means": [0] * 4,
        "scales": [1] * 4,
        "parameters": export_parameters(kind, model),
        "threshold": 0.5,
        "model_version": "fixture",
        "release": {"approved": False},
    }
    scorer = PortableModel(artifact)
    assert np.allclose(
        [scorer.score(dict(zip(FEATURE_NAMES, row))) for row in x], model.predict_proba(x)[:, 1]
    )
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(artifact))
    probability, decision = FraudScorer(path).predict(Transaction.model_validate(transaction))
    wanted = model.predict_proba([[transaction[name] for name in FEATURE_NAMES]])[0, 1]
    assert probability == pytest.approx(wanted) and decision == (wanted >= 0.5)


def test_comparison_writes_reproducible_selection_and_frozen_artifact(tmp_path):
    world = TransactionWorld(
        SimulationConfig(
            customers=20,
            terminals=5,
            terminal_compromise_rate=0.2,
            customer_compromise_rate=0.15,
            unusual_purchase_fraud_rate=0.05,
        ),
        parse_start("2024-01-01"),
    )
    records = pd.DataFrame([row for day in range(15) for row in world.day(day)])
    source = tmp_path / "features.csv"
    build_training_features(validate_and_sort(records)).to_csv(source, index=False)
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "seed": 7,
                "mlp_epochs": 1,
                "max_false_positive_rate": 0.1,
                "min_positive_per_split": 1,
                "min_negative_per_split": 1,
            }
        )
    )
    output = tmp_path / "comparison"
    report = run(source, output, policy)
    assert len(report["candidates"]) == 6
    frozen = json.loads((output / "selection-before-test.json").read_text())
    assert report["selected"] == frozen["selected"] == choose(frozen["validation"])
    assert not report["selection_uses_test"] and not report["production_approved"]
    assert all(
        candidate["validation"]["false_positive_rate"] <= 0.1
        for candidate in report["candidates"].values()
    )
    assert (output / "MODEL_SELECTION.md").exists()
    assert (output / "selected-model.sha256").exists()
    with pytest.raises(FileExistsError):
        run(source, output, policy)
    # A second fixed-seed run reproduces scores, thresholds and selection, not wall-clock timing.
    again = run(source, tmp_path / "second", policy)
    assert again["selected"] == report["selected"]
    for name in report["candidates"]:
        assert again["candidates"][name]["test"] == report["candidates"][name]["test"]


def test_cyclic_tree_and_missing_v2_inputs_fail_closed(tmp_path, transaction):
    from fraud_detector.features.handbook import V2_FEATURE_NAMES

    bad = {
        "format_version": 1,
        "feature_names": FEATURE_NAMES,
        "model_type": "hist_gradient_boosting",
        "means": [0] * 4,
        "scales": [1] * 4,
        "parameters": {
            "baseline": 0,
            "trees": [[{"feature": 0, "threshold": 1, "left": 0, "right": 0}]],
        },
    }
    with pytest.raises(ValueError, match="cyclic"):
        PortableModel(bad)
    good = {
        "format_version": 1,
        "feature_names": V2_FEATURE_NAMES,
        "model_type": "logistic",
        "means": [0] * len(V2_FEATURE_NAMES),
        "scales": [1] * len(V2_FEATURE_NAMES),
        "parameters": {"coefficients": [0] * len(V2_FEATURE_NAMES), "intercept": 0},
        "threshold": 0.5,
        "model_version": "fixture",
        "release": {"approved": False},
    }
    path = tmp_path / "v2.json"
    path.write_text(json.dumps(good))
    with pytest.raises(MissingBehavioralFeatures):
        FraudScorer(path).predict(Transaction.model_validate(transaction))
