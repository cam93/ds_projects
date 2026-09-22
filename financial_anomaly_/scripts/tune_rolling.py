"""Frozen rolling-window tuning and terminal-risk ablation; no deployment."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from fraud_detector.artifacts import digest
from fraud_detector.dataset import population
from fraud_detector.features.adaptive import V3_FEATURE_NAMES
from fraud_detector.features.terminal_risk import V4_FEATURE_NAMES, add_terminal_risk
from fraud_detector.model.candidates import FittedCandidate
from fraud_detector.model.evaluation import metrics, precision_curve


def operating_threshold(labels, scores, target):
    if not 0 < np.asarray(labels).sum() < len(labels):
        raise ValueError("Calibration requires both classes")
    thresholds, precision, recall = precision_curve(labels, scores)
    eligible = np.flatnonzero(recall >= target)
    best = max(eligible, key=lambda i: (precision[i], thresholds[i]))
    return float(thresholds[best])


def windows(frame, evaluation_start):
    # Days 0..14 provide warm-up history; labels require seven days to mature.
    day = frame.day
    return (
        frame[(day >= 14) & (day < evaluation_start - 28)],
        frame[(day >= evaluation_start - 21) & (day < evaluation_start - 7)],
        frame[(day >= evaluation_start) & (day < evaluation_start + 14)],
    )


def evaluate(frame, scores, threshold):
    result = metrics(frame, scores, threshold)
    result["false_negative_rate"] = 1 - result["recall"]
    result["alerts_per_1000"] = (
        1000 * (result["true_positive"] + result["false_positive"]) / len(frame)
    )
    return result


def summarize(folds):
    return {
        "minimum_recall": min(f["recall"] for f in folds),
        "mean_precision": float(np.mean([f["precision"] for f in folds])),
        "worst_alerts_per_1000": max(f["alerts_per_1000"] for f in folds),
        "mean_false_positive_rate": float(np.mean([f["false_positive_rate"] for f in folds])),
    }


def choose(results):
    eligible = [k for k, v in results.items() if v["summary"]["minimum_recall"] > 0.85]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda k: (
            results[k]["summary"]["mean_precision"],
            -results[k]["summary"]["worst_alerts_per_1000"],
            k,
        ),
    )


def fit(frame, names, parameters):
    model = HistGradientBoostingClassifier(**parameters).fit(
        frame[names].to_numpy(), frame.is_fraud
    )
    return FittedCandidate(
        model,
        names,
        "hist_gradient_boosting",
        dict.fromkeys(names, 0.0),
        dict.fromkeys(names, 1.0),
        parameters,
    )


def artifact(fitted, threshold, version):
    return {
        "format_version": 1,
        "feature_version": 3 if len(fitted.names) == 28 else 4,
        "feature_names": fitted.names,
        "feedback_delay_days": 7,
        "model_type": "hist_gradient_boosting",
        "model_version": version,
        "means": fitted.mean_vector,
        "scales": fitted.scale_vector,
        "parameters": fitted.parameters,
        "threshold": threshold,
        "release": {"approved": False, "purpose": "Offline experiment; no promotion"},
    }


def run(input_path, provenance_path, output):
    source_hash = digest(input_path)
    provenance = json.loads(provenance_path.read_text())
    if provenance["input_sha256"] != source_hash or provenance["label_delay_days"] != 7:
        raise ValueError("Expected verified seven-day-feedback features")
    output.mkdir(parents=True, exist_ok=False)
    config = {
        "fold_starts": [84, 112, 140],
        "recall_targets": [0.86, 0.90, 0.95],
        "feedback_delay_days": 7,
        "warmup_days": 14,
        "external_seeds": [2026092101, 2026092102],
        "external_start": "2026-01-01",
        "external_policy": {
            "days": 112,
            "customers": 500,
            "terminals": 50,
            "feedback_delay_days": 7,
        },
        "input_sha256": source_hash,
        "selection": "Every rolling evaluation must have recall >85%; maximize mean precision, then minimize worst alert volume. If none qualify, report failure without claiming a winner.",
        "external_limit": "Fresh local simulator populations, not new Handbook data. This is a cross-generator stress test, not a same-distribution final benchmark.",
    }
    search = [
        {
            "max_iter": i,
            "max_leaf_nodes": l,
            "min_samples_leaf": m,
            "learning_rate": r,
            "l2_regularization": reg,
            "class_weight": w,
            "early_stopping": False,
            "random_state": 7,
        }
        for i, l, m, r, reg, w in [
            (150, 15, 50, 0.08, 1, "balanced"),
            (150, 15, 50, 0.08, 1, None),
            (250, 15, 100, 0.05, 10, {0: 1, 1: 10}),
            (250, 7, 100, 0.05, 10, None),
            (250, 31, 50, 0.05, 10, {0: 1, 1: 10}),
            (250, 31, 100, 0.05, 10, "balanced"),
        ]
    ]
    config["search"] = search
    (output / "protocol.json").write_text(json.dumps(config, indent=2) + "\n")
    print("Preparing experimental point-in-time features", flush=True)
    frame = pd.read_csv(input_path, dtype={"transaction_id": str, "terminal_id": str})
    if frame.transaction_id.isna().any() or frame.transaction_id.duplicated().any():
        raise ValueError("Non-null unique IDs required")
    frame = add_terminal_risk(frame)
    if not np.isfinite(frame[V4_FEATURE_NAMES].to_numpy()).all():
        raise ValueError("Nonfinite features")
    frame["day"] = (frame.timestamp - frame.timestamp.min().normalize()).dt.total_seconds() / 86400
    results = {}
    with threadpool_limits(limits=1):
        for version, names in [("v3", V3_FEATURE_NAMES), ("v4", V4_FEATURE_NAMES)]:
            for index, parameters in enumerate(search):
                for start in config["fold_starts"]:
                    print(f"{version} configuration {index + 1}/6 fold {start}", flush=True)
                    training, calibration, validation = windows(frame, start)
                    for split in (training, calibration, validation):
                        if not 0 < split.is_fraud.sum() < len(split):
                            raise ValueError("Each window needs both classes")
                    fitted = fit(training, names, parameters)
                    cs, vs = fitted.score(calibration), fitted.score(validation)
                    for target in config["recall_targets"]:
                        key = f"{version}-{index}-recall-{target}"
                        threshold = operating_threshold(calibration.is_fraud, cs, target)
                        item = results.setdefault(
                            key,
                            {
                                "version": version,
                                "parameters": parameters,
                                "target": target,
                                "folds": [],
                            },
                        )
                        item["folds"].append(
                            {
                                "evaluation_start_day": start,
                                "threshold": threshold,
                                **evaluate(validation, vs, threshold),
                            }
                        )
                        item["summary"] = summarize(item["folds"])
                    (output / "progress.json").write_text(json.dumps(results, indent=2) + "\n")
        selected = choose(results)
        # Diagnostic fallback is explicitly not an eligible winner.
        diagnostic = selected or max(
            results,
            key=lambda k: (
                results[k]["summary"]["minimum_recall"],
                results[k]["summary"]["mean_precision"],
            ),
        )
        ablation = {
            v: choose({k: r for k, r in results.items() if r["version"] == v}) for v in ["v3", "v4"]
        }
        frozen = {
            **config,
            "selected": selected,
            "diagnostic_candidate": diagnostic,
            "ablation": ablation,
            "results": results,
            "deployed": False,
        }
        (output / "selection-before-final.json").write_text(json.dumps(frozen, indent=2) + "\n")
        # Final fit uses historical days 14..148, calibration 155..169; later known
        # Handbook data remains excluded. External populations generated only afterward.
        training, calibration, _ = windows(frame, 176)
        chosen = results[diagnostic]
        final_models = {}
        for key, parameters, names, target in [
            ("baseline", search[0], V3_FEATURE_NAMES, 0.86),
            (
                "candidate",
                chosen["parameters"],
                V3_FEATURE_NAMES if chosen["version"] == "v3" else V4_FEATURE_NAMES,
                chosen["target"],
            ),
        ]:
            fitted = fit(training, names, parameters)
            cs = fitted.score(calibration)
            threshold = operating_threshold(calibration.is_fraud, cs, target)
            exported = artifact(fitted, threshold, f"rolling-{key}-{source_hash[:12]}")
            parity = fitted.verify_export(calibration, cs, exported)
            path = output / f"{key}.json"
            path.write_text(json.dumps(exported) + "\n")
            final_models[key] = (fitted, threshold)
            frozen[key + "_artifact"] = {
                "sha256": digest(path),
                "threshold": threshold,
                "parity_error": parity,
            }
        (output / "artifacts-before-final.json").write_text(json.dumps(frozen, indent=2) + "\n")
        external = []
        for seed in config["external_seeds"]:
            print(f"Generating fresh stress-test population {seed}", flush=True)
            fresh, _ = population(config["external_policy"], seed, config["external_start"])
            fresh = add_terminal_risk(fresh)
            fresh = fresh[fresh.day >= 14]
            external.append(
                {
                    "seed": seed,
                    "rows": len(fresh),
                    **{
                        key: evaluate(fresh, fitted.score(fresh), threshold)
                        for key, (fitted, threshold) in final_models.items()
                    },
                }
            )
        frozen["external"] = external
        frozen["external_targets_passed"] = all(
            e["candidate"]["recall"] > 0.85
            and e["candidate"]["precision"] > e["baseline"]["precision"]
            for e in external
        )
        (output / "report.json").write_text(json.dumps(frozen, indent=2) + "\n")
        print(json.dumps({"selected": selected, "external": external}, indent=2), flush=True)
    return frozen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data/processed/handbook-full/training_features.csv")
    )
    parser.add_argument(
        "--provenance", type=Path, default=Path("reports/handbook-full/training.json")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.input, args.provenance, args.output_dir)
