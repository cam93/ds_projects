"""Compare 18/28-feature boosting on verified, seven-day-feedback Handbook features."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from fraud_detector.artifacts import digest
from fraud_detector.features.adaptive import V3_FEATURE_NAMES
from fraud_detector.features.handbook import V2_FEATURE_NAMES
from fraud_detector.model.candidates import fit_candidate
from fraud_detector.model.evaluation import metrics, precision_curve, split_by_time


def high_recall_threshold(labels, scores):
    if not 0 < np.asarray(labels).sum() < len(labels):
        raise ValueError("Validation requires both classes")
    thresholds, precision, recall = precision_curve(labels, scores)
    eligible = np.flatnonzero(recall > 0.85)
    best = max(eligible, key=lambda i: (precision[i], thresholds[i]))
    return float(thresholds[best])


def run(input_path, provenance_path, output):
    provenance = json.loads(provenance_path.read_text())
    source_hash = digest(input_path)
    if provenance["input_sha256"] != source_hash or provenance["label_delay_days"] != 7:
        raise ValueError("Input must match the recorded seven-day-feedback preparation run")
    output.mkdir(parents=True, exist_ok=False)
    data = pd.read_csv(input_path, dtype={"transaction_id": str})
    if data.transaction_id.isna().any() or data.transaction_id.duplicated().any():
        raise ValueError("Transaction IDs must be non-null and unique")
    if not np.isfinite(data[V3_FEATURE_NAMES].to_numpy(dtype=float)).all():
        raise ValueError("All 28 features must be finite")
    if not data.is_fraud.isin([0, 1]).all():
        raise ValueError("Labels must be binary")
    train, validation, test = split_by_time(data)
    train = train[train.timestamp < validation.timestamp.min() - pd.Timedelta(days=7)]
    validation = validation[validation.timestamp < test.timestamp.min() - pd.Timedelta(days=7)]
    splits = {}
    for name, frame in [("train", train), ("validation", validation), ("test", test)]:
        if not 0 < frame.is_fraud.sum() < len(frame):
            raise ValueError(f"{name} must contain both classes")
        splits[name] = {
            "rows": len(frame),
            "fraud": int(frame.is_fraud.sum()),
            "start": frame.timestamp.min().isoformat(),
            "end": frame.timestamp.max().isoformat(),
        }
    candidates, fitted_models = {}, {}
    with threadpool_limits(limits=1):
        for version, names in [(2, V2_FEATURE_NAMES), (3, V3_FEATURE_NAMES)]:
            key = f"v{version}"
            print(f"Training {len(names)} features", flush=True)
            fitted = fit_candidate(
                train, names, "hist_gradient_boosting", epochs=15, seed=7, hidden_size=8
            )
            scores = fitted.score(validation)
            threshold = high_recall_threshold(validation.is_fraud, scores)
            artifact = {
                "format_version": 1,
                "feature_version": version,
                "feature_names": names,
                "model_type": "hist_gradient_boosting",
                "feedback_delay_days": 7,
                "means": fitted.mean_vector,
                "scales": fitted.scale_vector,
                "parameters": fitted.parameters,
                "threshold": threshold,
                "model_version": f"handbook-{key}-hgb-{source_hash[:12]}",
                "release": {"approved": False, "purpose": "Offline comparison; no promotion"},
            }
            parity = fitted.verify_export(validation, scores, artifact)
            path = output / f"{key}-candidate.json"
            path.write_text(json.dumps(artifact) + "\n")
            candidates[key] = {
                "feature_count": len(names),
                "threshold": threshold,
                "validation": metrics(validation, scores, threshold),
                "hyperparameters": fitted.hyperparameters,
                "portable_max_abs_error": parity,
                "sha256": digest(path),
            }
            fitted_models[key] = fitted
        frozen = {
            "input_sha256": source_hash,
            "provenance_sha256": digest(provenance_path),
            "feedback_delay_days": 7,
            "splits": splits,
            "candidates": candidates,
            "selection": "Maximize validation precision subject to recall strictly above 85%",
            "evaluation_limit": "Test period previously evaluated in the 18-feature experiment; comparative evidence, not a fresh final holdout",
            "deployed": False,
        }
        (output / "selection-before-test.json").write_text(json.dumps(frozen, indent=2) + "\n")
        for key, fitted in fitted_models.items():
            result = metrics(test, fitted.score(test), candidates[key]["threshold"])
            result["false_negative_rate"] = 1 - result["recall"]
            result["alert_rate"] = (result["true_positive"] + result["false_positive"]) / len(test)
            candidates[key]["test"] = result
        frozen["recall_target_passed"] = candidates["v3"]["test"]["recall"] > 0.85
        frozen["precision_improved"] = (
            candidates["v3"]["test"]["precision"] > candidates["v2"]["test"]["precision"]
        )
        (output / "report.json").write_text(json.dumps(frozen, indent=2) + "\n")
    print(json.dumps({k: v["test"] for k, v in candidates.items()}, indent=2), flush=True)
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
