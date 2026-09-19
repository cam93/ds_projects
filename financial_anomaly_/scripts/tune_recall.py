"""Tune gradient boosting for high recall; preserve the deployed artifact until all gates pass."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from fraud_detector.artifacts import digest
from fraud_detector.dataset import population
from fraud_detector.features.handbook import V2_FEATURE_NAMES
from fraud_detector.model.candidates import FittedCandidate
from fraud_detector.model.evaluation import metrics, threshold_at_fpr
from fraud_detector.model.portable import PortableModel


def recall_threshold(labels, scores, target):
    labels, scores = np.asarray(labels), np.asarray(scores)
    if (
        labels.ndim != 1
        or scores.ndim != 1
        or len(labels) != len(scores)
        or not np.isin(labels, [0, 1]).all()
        or not np.isfinite(scores).all()
        or not 0 < target <= 1
        or not 0 < labels.sum() < len(labels)
    ):
        raise ValueError("Recall calibration requires both classes and a target in (0,1]")
    # Highest threshold detecting at least the target fraction, preserving score ties.
    fraud_scores = np.sort(scores[labels == 1])[::-1]
    return float(fraud_scores[int(np.ceil(target * len(fraud_scores))) - 1])


def evaluate(frame, scores, threshold):
    result = metrics(frame, scores, threshold)
    result["false_negative_rate"] = 1 - result["recall"]
    result["alert_rate"] = float((np.asarray(scores) >= threshold).mean())
    return result


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    policy = json.loads(Path("configs/robust-evaluation.json").read_text())
    names = V2_FEATURE_NAMES
    active_path = Path("models/artifacts/fraud_model.json")
    active = json.loads(active_path.read_text())
    baseline = PortableModel(active)
    search = [
        {
            "max_iter": iterations,
            "max_leaf_nodes": leaves,
            "min_samples_leaf": minimum,
            "learning_rate": rate,
            "l2_regularization": l2,
            "class_weight": weight,
            "early_stopping": False,
            "random_state": 7,
        }
        for iterations, leaves, minimum, rate, l2, weight in [
            (150, 15, 50, 0.08, 1.0, "balanced"),
            (300, 15, 100, 0.05, 10.0, "balanced"),
            (300, 31, 50, 0.05, 1.0, "balanced"),
            (300, 7, 100, 0.05, 10.0, "balanced"),
            (150, 15, 50, 0.08, 1.0, None),
            (300, 31, 100, 0.05, 10.0, None),
            (400, 31, 20, 0.05, 1.0, "balanced"),
            (300, 63, 100, 0.05, 10.0, "balanced"),
        ]
    ]
    print("Generating fresh training and tuning populations", flush=True)
    training, _ = population(policy, 3141)
    development, _ = population(policy, 3142)
    # Seven-day label delay respected; distinct later calibration and selection periods.
    train = training[training.day < 77]
    calibration = development[(development.day >= 84) & (development.day < 98)]
    validation = development[development.day >= 105]
    active_scores = np.array([baseline.score(row) for row in validation[names].to_dict("records")])
    baseline_validation = evaluate(validation, active_scores, active["threshold"])
    outcomes, fitted = {}, {}
    with threadpool_limits(limits=1):
        for index, parameters in enumerate(search):
            print(f"Fitting candidate {index + 1}/{len(search)}", flush=True)
            model = HistGradientBoostingClassifier(**parameters).fit(
                train[names].to_numpy(), train.is_fraud
            )
            candidate = FittedCandidate(
                model,
                names,
                "hist_gradient_boosting",
                dict.fromkeys(names, 0.0),
                dict.fromkeys(names, 1.0),
                parameters,
            )
            cs, vs = candidate.score(calibration), candidate.score(validation)
            for target in (0.86, 0.90, 0.95):
                key = f"candidate-{index}-recall-{target}"
                threshold = recall_threshold(calibration.is_fraud, cs, target)
                outcomes[key] = {
                    "parameters": parameters,
                    "calibration_recall_target": target,
                    "threshold": threshold,
                    "validation": evaluate(validation, vs, threshold),
                }
                fitted[key] = candidate
        # Prioritize recall and then precision; report FPR trade-off without hiding infeasibility.
        eligible = [k for k, v in outcomes.items() if v["validation"]["recall"] > 0.85]
        selected = max(
            eligible or outcomes,
            key=lambda k: (
                outcomes[k]["validation"]["precision"]
                if eligible
                else outcomes[k]["validation"]["recall"],
                outcomes[k]["validation"]["recall"],
            ),
        )
        chosen = outcomes[selected]
        candidate = fitted[selected]
        artifact = {
            "format_version": 1,
            "feature_names": names,
            "model_type": "hist_gradient_boosting",
            "means": candidate.mean_vector,
            "scales": candidate.scale_vector,
            "parameters": candidate.parameters,
            "threshold": chosen["threshold"],
            "model_version": "recall-tuned-"
            + hashlib.sha256(json.dumps(candidate.parameters, sort_keys=True).encode()).hexdigest()[
                :12
            ],
            "release": {"approved": False, "purpose": "high-recall experiment; review required"},
        }
        candidate.verify_export(validation, candidate.score(validation), artifact)
        frozen = {
            "selected": selected,
            "baseline_validation": baseline_validation,
            "candidates": outcomes,
            "selection": "highest validation precision among candidates with recall strictly above 85%; no FPR cap applied to this diagnostic selection",
            "seeds": {"training": 3141, "development": 3142, "final": 3143},
            "active_sha256": digest(active_path),
            "data_policy": policy,
            "split_days": {"training": [0, 77], "calibration": [84, 98], "validation": [105, 112]},
        }
        (output / "selection-before-test.json").write_text(json.dumps(frozen, indent=2) + "\n")
        (output / "candidate.json").write_text(json.dumps(artifact) + "\n")
        print("Selection frozen; evaluating fresh final population", flush=True)
        final, _ = population(policy, 3143, "2025-07-01")
        portable = PortableModel(artifact)
        rows = final[names].to_dict("records")
        scores = np.array([portable.score(row) for row in rows])
        active_scores = np.array([baseline.score(row) for row in rows])
        final_result = evaluate(final, scores, artifact["threshold"])
        baseline_result = evaluate(final, active_scores, active["threshold"])
        # Diagnostics for the current model use thresholds selected on calibration, never final labels.
        cal_active = np.array(
            [baseline.score(row) for row in calibration[names].to_dict("records")]
        )
        operating_points = {}
        for cap in (0.01, 0.05, 0.10):
            threshold = threshold_at_fpr(calibration.is_fraud, cal_active, cap)
            operating_points[str(cap)] = {
                "threshold": threshold,
                "final": evaluate(final, active_scores, threshold),
            }
        report = {
            **frozen,
            "final": final_result,
            "baseline_final": baseline_result,
            "baseline_fpr_operating_points": operating_points,
            "candidate_sha256": digest(output / "candidate.json"),
            "final_rows": len(final),
            "recall_gate_passed": final_result["recall"] > 0.85,
            "precision_improved": final_result["precision"] > baseline_result["precision"],
            "deployed": False,
        }
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    k: report[k]
                    for k in [
                        "selected",
                        "final",
                        "baseline_final",
                        "recall_gate_passed",
                        "precision_improved",
                    ]
                },
                indent=2,
            )
        )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args().output_dir)
