"""Reference baseline, common-feature model comparison, and fresh Handbook population."""

import argparse
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from fraud_detector.artifacts import digest
from fraud_detector.dataset import read_source
from fraud_detector.features.official_handbook import OFFICIAL_FEATURES, official_features
from fraud_detector.model.evaluation import classification_metrics
from scripts.handbook_reference.reference import (
    add_frauds,
    card_precision_top_k,
    generate_customer_profiles_table,
    generate_terminal_profiles_table,
    generate_transactions_table,
    get_count_risk_rolling_window,
    get_customer_spending_behaviour_features,
    get_list_terminals_within_radius,
    get_train_test_set,
)
from scripts.tune_rolling import operating_threshold


def assess(frame, scores, threshold):
    from sklearn.metrics import roc_auc_score

    result = classification_metrics(frame.TX_FRAUD, scores, threshold)
    result["rows"] = len(frame)
    result["false_negative_rate"] = 1 - result["recall"]
    result["roc_auc"] = float(roc_auc_score(frame.TX_FRAUD, scores))
    result["alerts_per_1000"] = float(np.mean(scores >= threshold) * 1000)
    predictions = frame[["CUSTOMER_ID", "TX_FRAUD", "TX_TIME_DAYS"]].copy()
    predictions["predictions"] = scores
    result["card_precision_at_100"] = float(card_precision_top_k(predictions, 100)[2])
    return result


def model(name):
    choices = {
        "decision_tree_2": lambda: DecisionTreeClassifier(max_depth=2, random_state=0),
        "logistic": lambda: LogisticRegression(random_state=0, max_iter=1000),
        "random_forest": lambda: RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=2),
        "xgboost": lambda: XGBClassifier(random_state=0, n_jobs=2),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(
            max_iter=150,
            max_leaf_nodes=15,
            min_samples_leaf=50,
            learning_rate=0.08,
            l2_regularization=1,
            class_weight="balanced",
            early_stopping=False,
            random_state=7,
        ),
        "hgb_unweighted": lambda: HistGradientBoostingClassifier(
            max_iter=250,
            max_leaf_nodes=15,
            min_samples_leaf=100,
            learning_rate=0.05,
            l2_regularization=10,
            early_stopping=False,
            random_state=7,
        ),
        "rf_regularized": lambda: RandomForestClassifier(
            n_estimators=150, min_samples_leaf=5, max_features=0.7, random_state=0, n_jobs=2
        ),
        "xgb_regularized": lambda: XGBClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            min_child_weight=5,
            reg_lambda=10,
            random_state=0,
            n_jobs=2,
        ),
    }
    return make_pipeline(StandardScaler(), choices[name]())


def period(frame, start, end):
    return frame[
        (frame.TX_DATETIME >= pd.Timestamp(start, tz="UTC"))
        & (frame.TX_DATETIME < pd.Timestamp(end, tz="UTC"))
    ]


def verify_reference(frame):
    errors = {}
    for entity, fn, names in [
        ("CUSTOMER_ID", get_customer_spending_behaviour_features, OFFICIAL_FEATURES[3:9]),
        ("TERMINAL_ID", get_count_risk_rolling_window, OFFICIAL_FEATURES[9:]),
    ]:
        # Select deterministic entities including the earliest fraud cases.
        ids = list(frame[entity].unique()[:10]) + list(
            frame.loc[frame.TX_FRAUD == 1, entity].unique()[:10]
        )
        largest = 0.0
        for identity in set(ids):
            group = frame[frame[entity] == identity].copy()
            reference = fn(group).set_index("TRANSACTION_ID")
            actual = group.set_index("TRANSACTION_ID")
            # Match by ID; reference sorting ties may differ for same-time customer averages.
            unique = ~group.TX_DATETIME.duplicated(keep=False)
            keys = group.loc[unique, "TRANSACTION_ID"]
            a, b = actual.loc[keys, names].to_numpy(), reference.loc[keys, names].to_numpy()
            if not np.allclose(a, b, atol=1e-7, rtol=1e-7):
                raise ValueError(f"Reference feature mismatch: {entity} {identity}")
            largest = max(largest, float(np.max(np.abs(a - b))))
        errors[entity] = largest
    return errors


def fresh_population(seed, days=112):
    # Original reference mechanism and sizes; new profile seeds and customer RNG IDs.
    customers = generate_customer_profiles_table(5000, random_state=seed)
    customers["CUSTOMER_ID"] += seed * 5000
    terminals = generate_terminal_profiles_table(10000, random_state=seed + 1)
    xy = terminals[["x_terminal_id", "y_terminal_id"]].to_numpy()
    customers["available_terminals"] = customers.apply(
        lambda row: get_list_terminals_within_radius(row, xy, 5), axis=1
    )
    frames = [
        generate_transactions_table(row, start_date="2026-01-01", nb_days=days)
        for row in customers.itertuples(index=False)
    ]
    data = pd.concat(frames, ignore_index=True).sort_values("TX_DATETIME").reset_index(drop=True)
    data.insert(0, "TRANSACTION_ID", np.arange(len(data)))
    return official_features(add_frauds(customers, terminals, data))


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = Path("configs/trusted-handbook-6e67dbd.json")
    manifest = json.loads(manifest_path.read_text())
    protocol = {
        "raw_commit": "6e67dbd0a3bfe0d7ec33abc4bce5f37cd4ff0d6a",
        "reference_commit": "81cf7d1714bb7b2f5b496407d9055d91dc68dc25",
        "reference_sha256": digest(Path("scripts/handbook_reference/reference.py")),
        "manifest_sha256": digest(manifest_path),
        "features": OFFICIAL_FEATURES,
        "versions": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "fresh_seed": 941,
        "fresh_days": 112,
        "warmup_days": 37,
        "recall_calibration_target": 0.86,
        "deployed": False,
        "application_note": "All positive-amount transactions; no known-card exclusions. Same reference feature conventions. Customer windows include the current observed amount.",
        "reference_note": "Includes zero amounts to reproduce published data; default library versions differ from the historical notebook.",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print("Loading verified raw data and computing full Handbook features", flush=True)
    raw = pd.concat(
        [read_source(Path("data/raw") / name, manifest) for name in sorted(manifest)],
        ignore_index=True,
    )
    data = official_features(raw)
    del raw
    protocol["reference_feature_parity"] = verify_reference(data)
    training, official_test = get_train_test_set(data, pd.Timestamp("2018-07-25", tz="UTC"))
    all_test = period(data, "2018-08-08", "2018-08-15")
    reference = {}
    with threadpool_limits(limits=2):
        for name in [
            "decision_tree_2",
            "logistic",
            "random_forest",
            "xgboost",
            "hist_gradient_boosting",
        ]:
            print("Reference baseline:", name, flush=True)
            fitted = model(name).fit(training[OFFICIAL_FEATURES], training.TX_FRAUD)
            reference[name] = {
                "official": assess(
                    official_test, fitted.predict_proba(official_test[OFFICIAL_FEATURES])[:, 1], 0.5
                ),
                "all_transactions": assess(
                    all_test, fitted.predict_proba(all_test[OFFICIAL_FEATURES])[:, 1], 0.5
                ),
            }
        protocol["reference"] = reference
        protocol["published_reference"] = {
            "decision_tree_2": {
                "roc_auc": 0.763,
                "average_precision": 0.496,
                "card_precision_at_100": 0.241,
            },
            "random_forest": {
                "roc_auc": 0.867,
                "average_precision": 0.658,
                "card_precision_at_100": 0.287,
            },
            "xgboost": {
                "roc_auc": 0.862,
                "average_precision": 0.639,
                "card_precision_at_100": 0.273,
            },
        }
        (output / "reference.json").write_text(json.dumps(protocol, indent=2) + "\n")
        # Rolling comparison/calibration is separate from the fixed published baseline.
        dates = [
            ("2018-05-01", "2018-05-29", "2018-06-05", "2018-06-19", "2018-06-26", "2018-07-10"),
            ("2018-06-01", "2018-06-29", "2018-07-06", "2018-07-20", "2018-07-27", "2018-08-10"),
        ]
        protocol["rolling_windows"] = dates
        results = {}
        for name in [
            "hist_gradient_boosting",
            "random_forest",
            "xgboost",
            "hgb_unweighted",
            "rf_regularized",
            "xgb_regularized",
        ]:
            results[name] = []
            for bounds in dates:
                print("Rolling comparison:", name, bounds[0], flush=True)
                train, cal, val = [period(data, *bounds[i : i + 2]) for i in (0, 2, 4)]
                train, cal, val = [x[x.TX_AMOUNT > 0] for x in [train, cal, val]]
                fitted = model(name).fit(train[OFFICIAL_FEATURES], train.TX_FRAUD)
                threshold = operating_threshold(
                    cal.TX_FRAUD, fitted.predict_proba(cal[OFFICIAL_FEATURES])[:, 1], 0.86
                )
                results[name].append(
                    {
                        "threshold": threshold,
                        **assess(
                            val, fitted.predict_proba(val[OFFICIAL_FEATURES])[:, 1], threshold
                        ),
                    }
                )
            (output / "rolling-progress.json").write_text(json.dumps(results, indent=2) + "\n")
        eligible = [k for k, v in results.items() if min(f["recall"] for f in v) > 0.85]
        selected = (
            max(eligible, key=lambda k: np.mean([f["precision"] for f in results[k]]))
            if eligible
            else None
        )
        diagnostic = selected or max(
            results,
            key=lambda k: (
                min(f["recall"] for f in results[k]),
                np.mean([f["precision"] for f in results[k]]),
            ),
        )
        frozen = {
            **protocol,
            "rolling": results,
            "selected": selected,
            "diagnostic_candidate": diagnostic,
        }
        (output / "selection-before-final.json").write_text(json.dumps(frozen, indent=2) + "\n")
        train = period(data, "2018-08-01", "2018-08-29")
        cal = period(data, "2018-09-05", "2018-09-19")
        train = train[train.TX_AMOUNT > 0]
        cal = cal[cal.TX_AMOUNT > 0]
        models = {}
        for name in {diagnostic, "hist_gradient_boosting"}:
            fitted = model(name).fit(train[OFFICIAL_FEATURES], train.TX_FRAUD)
            threshold = operating_threshold(
                cal.TX_FRAUD, fitted.predict_proba(cal[OFFICIAL_FEATURES])[:, 1], 0.86
            )
            models[name] = (fitted, threshold)
        artifacts = Path("models/artifacts") / output.name
        artifacts.mkdir(exist_ok=False)
        for name, (fitted, threshold) in models.items():
            path = artifacts / f"{name}.joblib"
            joblib.dump(fitted, path)
            frozen.setdefault("artifacts", {})[name] = {
                "path": str(path),
                "sha256": digest(path),
                "threshold": threshold,
                "offline_only": True,
            }
        (output / "artifacts-before-final.json").write_text(json.dumps(frozen, indent=2) + "\n")
        print("Generating fresh original-Handbook population", flush=True)
        fresh = fresh_population(protocol["fresh_seed"], protocol["fresh_days"])
        evaluation = fresh[(fresh.TX_TIME_DAYS >= 37) & (fresh.TX_AMOUNT > 0)]
        frozen["fresh"] = {
            name: assess(
                evaluation, fitted.predict_proba(evaluation[OFFICIAL_FEATURES])[:, 1], threshold
            )
            for name, (fitted, threshold) in models.items()
        }
        frozen["fresh_rows"] = len(evaluation)
        frozen["fresh_target_passed"] = frozen["fresh"][diagnostic]["recall"] > 0.85
        (output / "report.json").write_text(json.dumps(frozen, indent=2) + "\n")
        print(json.dumps({"selected": selected, "fresh": frozen["fresh"]}, indent=2), flush=True)
    return frozen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args().output_dir)
