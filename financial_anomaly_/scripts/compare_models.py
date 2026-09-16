"""Frozen, chronological comparison: choose on validation, evaluate on test only afterward."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from train import (
    as_tensor,
    classification_metrics,
    fit_normalization,
    score_batches,
    split_by_time,
    train_model,
)

from fraud_detector.features.handbook import FEATURE_NAMES, FEATURE_VERSION, V2_FEATURE_NAMES
from fraud_detector.model.portable import PortableModel


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def threshold_at_fpr(labels, scores, max_fpr):
    y, s = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    if (
        not 0 <= max_fpr < 1
        or not len(y)
        or len(y) != len(s)
        or not np.isfinite(s).all()
        or not np.isin(y, [0, 1]).all()
        or not 0 < y.sum() < len(y)
    ):
        raise ValueError(
            "Threshold selection needs finite aligned scores, both classes and FPR in [0,1)"
        )
    order = np.argsort(-s, kind="stable")
    y, s = y[order], s[order]
    ends = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp = np.cumsum(y)[ends]
    fp = ends + 1 - tp
    eligible = np.flatnonzero(fp / (len(y) - y.sum()) <= max_fpr)
    if not len(eligible):
        return 2.0  # Explicit abstention: no probability can exceed this threshold.
    # Highest recall under the cap; for equal recall minimize false positives.
    chosen = max(eligible, key=lambda i: (tp[i], -fp[i]))
    return float(s[ends[chosen]])


def assess(records, scores, threshold):
    labels = records.is_fraud.to_numpy(dtype=int)
    result = classification_metrics(labels, scores, threshold)
    result["roc_auc"] = float(roc_auc_score(labels, scores))
    predicted = np.asarray(scores) >= threshold
    scenarios = {}
    for scenario in sorted(records.fraud_scenario.unique()):
        selected = (records.fraud_scenario.to_numpy() == scenario) & (labels == 1)
        total = int(selected.sum())
        if total:
            detected = int((predicted & selected).sum())
            scenarios[str(int(scenario))] = {
                "fraud_cases": total,
                "detected": detected,
                "missed": total - detected,
                "recall": detected / total,
            }
    result["scenarios"] = scenarios
    ranked = records[["timestamp", "transaction_id", "is_fraud"]].copy()
    ranked["score"] = scores
    ranked["day"] = pd.to_datetime(ranked.timestamp, utc=True).dt.date
    daily = (
        ranked.sort_values(["score", "transaction_id"], ascending=[False, True], kind="mergesort")
        .groupby("day")
        .head(100)
    )
    result["mean_daily_precision_at_100"] = float(daily.groupby("day").is_fraud.mean().mean())
    return result


def export_parameters(kind, model):
    if kind == "logistic":
        return {"coefficients": model.coef_[0].tolist(), "intercept": float(model.intercept_[0])}
    if kind == "mlp":
        return {
            "hidden_weights": model[0].weight.detach().tolist(),
            "hidden_bias": model[0].bias.detach().tolist(),
            "output_weights": model[2].weight.detach()[0].tolist(),
            "output_bias": float(model[2].bias.detach()[0]),
        }
    trees = []
    # Private sklearn representation is pinned and verified against native predict_proba.
    # Only numeric finite features are allowed: no categorical/missing-value routing.
    for predictors in model._predictors:
        if len(predictors) != 1:
            raise ValueError("Only binary numeric boosting models are supported")
        nodes = []
        for node in predictors[0].nodes:
            if node["is_leaf"]:
                nodes.append({"value": float(node["value"])})
            else:
                if node["is_categorical"]:
                    raise ValueError("Categorical trees are unsupported")
                nodes.append(
                    {
                        "feature": int(node["feature_idx"]),
                        "threshold": float(node["num_threshold"]),
                        "left": int(node["left"]),
                        "right": int(node["right"]),
                    }
                )
        trees.append(nodes)
    return {"baseline": float(model._baseline_prediction[0, 0]), "trees": trees}


def read_features(path):
    data = pd.read_csv(path, dtype={"transaction_id": str})
    if data.empty or data.transaction_id.isna().any() or data.transaction_id.duplicated().any():
        raise ValueError("Non-empty unique transaction IDs required")
    if not data.is_fraud.isin([0, 1]).all() or not data.fraud_scenario.isin([0, 1, 2, 3]).all():
        raise ValueError("Invalid ground-truth labels/scenarios")
    if not np.isfinite(data[V2_FEATURE_NAMES].to_numpy(dtype=float)).all():
        raise ValueError("All v2 features must be present and finite")
    data["timestamp"] = pd.to_datetime(data.timestamp, utc=True, errors="raise")
    return data


def choose(validation_results):
    # This interface deliberately has no test/external data argument.
    return max(
        validation_results,
        key=lambda name: (
            validation_results[name]["recall"],
            validation_results[name]["precision"],
            validation_results[name]["average_precision"],
            name,
        ),
    )


def run(input_path, output_dir, policy_path, external_input=None):
    policy = json.loads(Path(policy_path).read_text())
    expected = {
        "seed",
        "mlp_epochs",
        "max_false_positive_rate",
        "min_positive_per_split",
        "min_negative_per_split",
    }
    if set(policy) != expected:
        raise ValueError("Comparison policy has missing or unknown fields")
    if any(
        type(policy[key]) is not int or policy[key] < (0 if key == "seed" else 1)
        for key in ("seed", "mlp_epochs", "min_positive_per_split", "min_negative_per_split")
    ):
        raise ValueError("Policy counts/seed must be valid integers")
    if (
        not isinstance(policy["max_false_positive_rate"], (int, float))
        or not 0 <= policy["max_false_positive_rate"] < 1
    ):
        raise ValueError("Invalid false-positive-rate cap")
    output = Path(output_dir)
    output.mkdir(
        parents=True, exist_ok=False
    )  # Immutable run directory, no silent retuning overwrite.
    records = read_features(input_path)
    training, validation, test = split_by_time(records)
    splits = {}
    for name, frame in zip(("train", "validation", "test"), (training, validation, test)):
        positives = int(frame.is_fraud.sum())
        splits[name] = {
            "rows": len(frame),
            "fraud": positives,
            "genuine": len(frame) - positives,
            "start": frame.timestamp.min().isoformat(),
            "end": frame.timestamp.max().isoformat(),
        }
        if (
            positives < policy["min_positive_per_split"]
            or len(frame) - positives < policy["min_negative_per_split"]
        ):
            raise ValueError(f"Insufficient {name} class counts: {splits[name]}")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    candidates, native_models, validation_results = {}, {}, {}
    hashes = {"input_sha256": digest(input_path), "policy_sha256": digest(policy_path)}
    with threadpool_limits(limits=1):
        for feature_version, names in (("v1", FEATURE_NAMES), ("v2", V2_FEATURE_NAMES)):
            means, stds = fit_normalization(training, names)
            mean_vector, scale_vector = [means[n] for n in names], [stds[n] for n in names]
            for kind in ("logistic", "hist_gradient_boosting", "mlp"):
                key = f"{feature_version}_{kind}"
                print(f"Training {key}", flush=True)
                started = time.perf_counter()
                scaled = kind != "hist_gradient_boosting"
                model_means = mean_vector if scaled else [0.0] * len(names)
                model_scales = scale_vector if scaled else [1.0] * len(names)
                x = (training[names].to_numpy(dtype=float) - model_means) / model_scales
                vx = (validation[names].to_numpy(dtype=float) - model_means) / model_scales
                if kind == "mlp":
                    model, _, _ = train_model(
                        training,
                        validation,
                        means,
                        stds,
                        policy["mlp_epochs"],
                        names,
                        seed=policy["seed"],
                        hidden_size=8 if feature_version == "v1" else 32,
                    )
                    scores = np.asarray(
                        score_batches(model, as_tensor(validation, means, stds, names)[0])
                    )
                    hyperparameters = {
                        "epochs": policy["mlp_epochs"],
                        "hidden_size": 8 if feature_version == "v1" else 32,
                        "batch_size": 1024,
                        "learning_rate": 0.001,
                        "positive_class_weight": "balanced",
                    }
                else:
                    if kind == "logistic":
                        model = LogisticRegression(
                            C=1.0,
                            class_weight="balanced",
                            max_iter=2000,
                            random_state=policy["seed"],
                        )
                    else:
                        model = HistGradientBoostingClassifier(
                            max_iter=150,
                            max_leaf_nodes=15,
                            min_samples_leaf=50,
                            learning_rate=0.08,
                            l2_regularization=1.0,
                            class_weight="balanced",
                            early_stopping=False,
                            random_state=policy["seed"],
                        )
                    model.fit(x, training.is_fraud)
                    scores = model.predict_proba(vx)[:, 1]
                    hyperparameters = model.get_params()
                threshold = threshold_at_fpr(
                    validation.is_fraud, scores, policy["max_false_positive_rate"]
                )
                artifact = {
                    "format_version": 1,
                    "feature_version": 1 if feature_version == "v1" else FEATURE_VERSION,
                    "feature_names": names,
                    "model_type": kind,
                    "means": model_means,
                    "scales": model_scales,
                    "parameters": export_parameters(kind, model),
                    "threshold": threshold,
                    "model_version": key + "-" + hashes["input_sha256"][:12],
                    "release": {
                        "approved": False,
                        "purpose": "synthetic-demo comparison; separate production approval required",
                    },
                }
                scorer = PortableModel(artifact)
                sample = validation.iloc[
                    np.linspace(0, len(validation) - 1, min(1000, len(validation)), dtype=int)
                ]
                predicted = np.array(
                    [scorer.score(row) for row in sample[names].to_dict("records")]
                )
                sample_indices = np.linspace(
                    0, len(validation) - 1, min(1000, len(validation)), dtype=int
                )
                if not np.allclose(predicted, scores[sample_indices], rtol=1e-5, atol=1e-6):
                    raise ValueError(f"Portable/native scoring mismatch: {key}")
                validation_results[key] = assess(validation, scores, threshold)
                candidates[key] = {
                    "feature_count": len(names),
                    "threshold": threshold,
                    "hyperparameters": hyperparameters,
                    "training_seconds": time.perf_counter() - started,
                    "validation": validation_results[key],
                    "portable_validation_max_abs_error": float(
                        np.max(np.abs(predicted - scores[sample_indices]))
                    ),
                }
                native_models[key] = (
                    model,
                    names,
                    model_means,
                    model_scales,
                    means,
                    stds,
                    artifact,
                )
        selected = choose(validation_results)
        frozen = {
            "selected": selected,
            "selection": "maximize validation recall under FPR cap; break ties by precision, AP, then name",
            "policy": policy,
            **hashes,
            "validation": validation_results,
        }
        (output / "selection-before-test.json").write_text(json.dumps(frozen, indent=2) + "\n")
        # Nothing below this point can alter model choice, fitted weights or thresholds.
        external = read_features(external_input) if external_input else None
        if external is not None and (
            digest(external_input) == hashes["input_sha256"]
            or not 0 < external.is_fraud.sum() < len(external)
        ):
            raise ValueError("External evaluation requires distinct data with both classes")
        for key, (model, names, mv, sv, means, stds, artifact) in native_models.items():
            for label, frame in (("test", test), ("external", external)):
                if frame is None:
                    continue
                if artifact["model_type"] == "mlp":
                    scores = score_batches(model, as_tensor(frame, means, stds, names)[0])
                else:
                    scores = model.predict_proba((frame[names].to_numpy(dtype=float) - mv) / sv)[
                        :, 1
                    ]
                candidates[key][label] = assess(frame, scores, artifact["threshold"])
                if label == "test":
                    pd.DataFrame(
                        {
                            "transaction_id": frame.transaction_id,
                            "timestamp": frame.timestamp,
                            "label": frame.is_fraud,
                            "scenario": frame.fraud_scenario,
                            "score": scores,
                            "predicted": np.asarray(scores) >= artifact["threshold"],
                        }
                    ).to_csv(output / f"{key}-test-predictions.csv", index=False)
        artifact = native_models[selected][-1]
        # Include fitted parameters in the identity, not just the input dataset hash.
        artifact["model_version"] += (
            "-"
            + hashlib.sha256(
                json.dumps(artifact["parameters"], sort_keys=True).encode()
            ).hexdigest()[:12]
        )
        model_path = output / "selected-model.json"
        model_path.write_text(json.dumps(artifact, separators=(",", ":"), allow_nan=False) + "\n")
        (output / "selected-model.sha256").write_text(digest(model_path) + "\n")
    report = {
        "synthetic_data": True,
        "selected": selected,
        "policy": policy,
        **hashes,
        "splits": splits,
        "feature_schemas": {"v1": FEATURE_NAMES, "v2": V2_FEATURE_NAMES},
        "candidates": candidates,
        "selection_uses_test": False,
        "production_approved": False,
        "selected_test_fpr_target_met": candidates[selected]["test"]["false_positive_rate"]
        <= policy["max_false_positive_rate"],
        "external": {
            "rows": len(external),
            "fraud": int(external.is_fraud.sum()),
            "sha256": digest(external_input),
        }
        if external is not None
        else None,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "torch": str(torch.__version__),
        },
    }
    (output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "MODEL_SELECTION.md").write_text(markdown_report(report))
    return report


def markdown_report(report):
    chosen = report["selected"]
    lines = [
        "# Synthetic fraud model selection",
        "",
        f"Selected candidate: **{chosen}**.",
        "",
        "Selection maximizes validation recall under the predeclared false-positive cap. Ties use precision, then average precision. Test and external labels never select a model or threshold.",
        "",
        "| Candidate | Features | Validation recall | Test precision | Test recall | Test FPR | Test AP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, candidate in report["candidates"].items():
        test = candidate["test"]
        lines.append(
            f"| {key} | {candidate['feature_count']} | {candidate['validation']['recall']:.3f} | {test['precision']:.3f} | {test['recall']:.3f} | {test['false_positive_rate']:.3%} | {test['average_precision']:.3f} |"
        )
    lines += [
        "",
        f"Selected model meets the {report['policy']['max_false_positive_rate']:.1%} FPR target on test: **{report['selected_test_fpr_target_met']}**. This is an observed result, not a guarantee on new traffic.",
        "",
        "## Selected model by fraud scenario",
        "",
        "| Scenario | Fraud cases | Detected | Missed | Recall |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for scenario, result in report["candidates"][chosen]["test"]["scenarios"].items():
        lines.append(
            f"| {scenario} | {result['fraud_cases']} | {result['detected']} | {result['missed']} | {result['recall']:.3f} |"
        )
    if report["external"]:
        lines += [
            "",
            "## Independent-population evaluation",
            "",
            "These additional data are evaluated only after selection, with frozen models and thresholds. No refitting or threshold adjustment occurs.",
            "",
            "| Candidate | Precision | Recall | FPR | AP |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for key, candidate in report["candidates"].items():
            result = candidate["external"]
            lines.append(
                f"| {key} | {result['precision']:.3f} | {result['recall']:.3f} | {result['false_positive_rate']:.3%} | {result['average_precision']:.3f} |"
            )
    lines += [
        "",
        "## Interpretation and limitations",
        "",
        "- Scenario 1: unusual purchases; scenario 2: compromised terminals; scenario 3: customer spending bursts.",
        "- Features use only prior unlabeled observations. Labels, scenario codes and IDs are excluded from model inputs; scaling is fit on training only.",
        "- Terminal fraud can look identical to legitimate purchases in this generator. Label-free terminal activity does not reveal which terminal was secretly compromised. Perfect recall is not a realistic goal.",
        "- The neural networks use 8 hidden units (v1) or 32 (v2); tree and logistic settings are fixed across feature sets. The MLP feature ablation also changes capacity and is not a pure feature-only experiment.",
        "- Existing test-period data were previously used for an earlier candidate report. The additional seeded population provides a fresh check; neither synthetic set establishes real-world performance.",
        "- Class weighting changes score calibration. Scores are ranking signals, not demonstrated real-world fraud probabilities.",
        "- selected-model.json is a data-only portable artifact, with native/portable parity checked. It is not production-approved or automatically deployed. Existing production release gates are unchanged.",
        "- comparison.json records split boundaries, confusion counts, scenario recall, daily precision@100, runtime, hyperparameters, dependency versions and input/policy hashes. selection-before-test.json freezes the choice before test scoring.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--external-input", type=Path)
    parser.add_argument("--policy", type=Path, default=Path("configs/model-comparison.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.input, args.output_dir, args.policy, args.external_input)
    print(
        json.dumps(
            {"selected": report["selected"], "report": str(args.output_dir / "MODEL_SELECTION.md")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
