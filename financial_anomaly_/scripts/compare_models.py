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
from threadpoolctl import threadpool_limits

from fraud_detector.artifacts import digest
from fraud_detector.features.handbook import FEATURE_NAMES, FEATURE_VERSION, V2_FEATURE_NAMES
from fraud_detector.model.candidates import fit_candidate
from fraud_detector.model.evaluation import assess, split_by_time, threshold_at_fpr
from fraud_detector.model.export import export_parameters


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
            for kind in ("logistic", "hist_gradient_boosting", "mlp"):
                key = f"{feature_version}_{kind}"
                print(f"Training {key}", flush=True)
                started = time.perf_counter()
                fitted = fit_candidate(
                    training,
                    names,
                    kind,
                    epochs=policy["mlp_epochs"],
                    seed=policy["seed"],
                    hidden_size=8 if feature_version == "v1" else 32,
                )
                scores = fitted.score(validation)
                threshold = threshold_at_fpr(
                    validation.is_fraud, scores, policy["max_false_positive_rate"]
                )
                artifact = {
                    "format_version": 1,
                    "feature_version": 1 if feature_version == "v1" else FEATURE_VERSION,
                    "feature_names": names,
                    "model_type": kind,
                    "means": fitted.mean_vector,
                    "scales": fitted.scale_vector,
                    "parameters": fitted.parameters,
                    "threshold": threshold,
                    "model_version": key + "-" + hashes["input_sha256"][:12],
                    "release": {
                        "approved": False,
                        "purpose": "synthetic-demo comparison; separate production approval required",
                    },
                }
                parity_error = fitted.verify_export(validation, scores, artifact)
                validation_results[key] = assess(validation, scores, threshold)
                candidates[key] = {
                    "feature_count": len(names),
                    "threshold": threshold,
                    "hyperparameters": fitted.hyperparameters,
                    "training_seconds": time.perf_counter() - started,
                    "validation": validation_results[key],
                    "portable_validation_max_abs_error": parity_error,
                }
                native_models[key] = (fitted, artifact)
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
        for key, (fitted, artifact) in native_models.items():
            for label, frame in (("test", test), ("external", external)):
                if frame is None:
                    continue
                scores = fitted.score(frame)
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


__all__ = [
    "assess",
    "choose",
    "digest",
    "export_parameters",
    "main",
    "markdown_report",
    "read_features",
    "run",
    "threshold_at_fpr",
]


if __name__ == "__main__":
    main()
