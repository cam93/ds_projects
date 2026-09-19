"""Rolling validation, conservative thresholds and a locked fresh-population final test."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from threadpoolctl import threadpool_limits

from fraud_detector.artifacts import digest
from fraud_detector.dataset import population
from fraud_detector.features.adaptive import V3_FEATURE_NAMES
from fraud_detector.features.handbook import V2_FEATURE_NAMES
from fraud_detector.model.candidates import fit_candidate
from fraud_detector.model.evaluation import metrics, threshold_at_fpr, wilson
from fraud_detector.model.portable import PortableModel
from fraud_detector.simulation import SimulationConfig


def split_window(frame, train_end, policy):
    delay, duration = policy["feedback_delay_days"], policy["calibration_days"]
    # Fit only labels already investigated when fitting starts. Calibrate after the same delay.
    train = frame[frame.day < train_end - delay]
    calibration = frame[(frame.day >= train_end) & (frame.day < train_end + duration)]
    start = train_end + duration + delay
    evaluation = frame[(frame.day >= start) & (frame.day < start + policy["evaluation_days"])]
    for subset in (train, calibration, evaluation):
        if subset.empty or subset.is_fraud.nunique() != 2:
            raise ValueError("Every rolling partition must contain both classes")
    return train, calibration, evaluation


def fit(train, calibration, names, kind, policy):
    fitted = fit_candidate(
        train, names, kind, epochs=policy["epochs"], seed=policy["seed"], hidden_size=32
    )
    artifact = {
        "format_version": 1,
        "feature_names": names,
        "model_type": kind,
        "means": fitted.mean_vector,
        "scales": fitted.scale_vector,
        "parameters": fitted.parameters,
        "feedback_delay_days": policy["feedback_delay_days"],
        "threshold": 2.0,
        "model_version": "unselected",
        "release": {"approved": False},
    }
    native = fitted.score(calibration)
    fitted.verify_export(calibration, native, artifact)
    return artifact, fitted.score, native


def select(results, max_fpr):
    summary = {}
    for key, folds in results.items():
        upper = max(f["metrics"]["fpr_95_interval"][1] for f in folds)
        summary[key] = {
            "mean_recall": float(np.mean([f["metrics"]["recall"] for f in folds])),
            "mean_precision": float(np.mean([f["metrics"]["precision"] for f in folds])),
            "max_fpr": max(f["metrics"]["false_positive_rate"] for f in folds),
            "worst_fpr_95_upper": upper,
            "fpr_constraint_met": upper <= max_fpr,
        }
    eligible = [key for key, row in summary.items() if row["fpr_constraint_met"]]
    if eligible:
        winner = max(
            eligible, key=lambda k: (summary[k]["mean_recall"], summary[k]["mean_precision"], k)
        )
    else:
        winner = min(
            summary, key=lambda k: (summary[k]["worst_fpr_95_upper"], -summary[k]["mean_recall"], k)
        )
    return winner, summary, bool(eligible)


def validate_policy(p):
    expected = {
        "training_seeds",
        "final_seed",
        "days",
        "customers",
        "terminals",
        "feedback_delay_days",
        "train_end_days",
        "calibration_days",
        "evaluation_days",
        "final_train_end_day",
        "fpr_targets",
        "max_fpr",
        "epochs",
        "seed",
    }
    if set(p) != expected:
        raise ValueError("Unexpected or missing policy fields")
    for key in (
        "days",
        "customers",
        "terminals",
        "calibration_days",
        "evaluation_days",
        "final_train_end_day",
        "epochs",
    ):
        if type(p[key]) is not int or p[key] < 1:
            raise ValueError("Policy dimensions must be positive integers")
    if (
        len(p["training_seeds"]) < 2
        or len(set(p["training_seeds"])) != len(p["training_seeds"])
        or p["final_seed"] in p["training_seeds"]
    ):
        raise ValueError("Final population must be distinct from multiple training populations")
    for seed in p["training_seeds"] + [p["final_seed"], p["seed"]]:
        SimulationConfig(seed=seed)
    if not 1 <= p["feedback_delay_days"] <= 30 or not 0 < p["max_fpr"] < 1:
        raise ValueError("Invalid delay or FPR limit")
    if not p["fpr_targets"] or any(not 0 < t <= p["max_fpr"] for t in p["fpr_targets"]):
        raise ValueError("Invalid threshold targets")
    if len(p["train_end_days"]) < 2 or len(set(p["train_end_days"])) != len(p["train_end_days"]):
        raise ValueError("At least two distinct rolling windows required")
    for end in p["train_end_days"]:
        if (
            end <= p["feedback_delay_days"]
            or end + p["calibration_days"] + p["feedback_delay_days"] + p["evaluation_days"]
            > p["days"]
        ):
            raise ValueError("Rolling window exceeds population dates")
    if (
        p["final_train_end_day"] <= p["feedback_delay_days"]
        or p["final_train_end_day"] + p["calibration_days"] + p["feedback_delay_days"] > p["days"]
    ):
        raise ValueError("Final calibration labels not available within population dates")


def run(policy_path, output):
    policy = json.loads(Path(policy_path).read_text())
    validate_policy(policy)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    results, populations, provenance = {}, [], []
    candidates = {
        f"{version}_{kind}": (names, kind)
        for version, names in [("v2", V2_FEATURE_NAMES), ("v3", V3_FEATURE_NAMES)]
        for kind in ("logistic", "hist_gradient_boosting", "mlp")
    }
    with threadpool_limits(limits=1):
        for seed in policy["training_seeds"]:
            print(f"Generating development population {seed}", flush=True)
            frame, config = population(policy, seed)
            populations.append(frame)
            provenance.append(
                {
                    "config": config,
                    "rows": len(frame),
                    "feature_sha256": hashlib.sha256(
                        frame.to_csv(index=False).encode()
                    ).hexdigest(),
                }
            )
            for end in policy["train_end_days"]:
                train, calibration, evaluation = split_window(frame, end, policy)
                for name, (names, kind) in candidates.items():
                    print(f"Rolling seed={seed} train_end={end} candidate={name}", flush=True)
                    _, score, cs = fit(train, calibration, names, kind, policy)
                    scores = score(evaluation)
                    for target in policy["fpr_targets"]:
                        threshold = threshold_at_fpr(calibration.is_fraud, cs, target)
                        key = f"{name}@{target}"
                        results.setdefault(key, []).append(
                            {
                                "seed": seed,
                                "train_end_day": end,
                                "threshold": threshold,
                                "train_rows": len(train),
                                "calibration_rows": len(calibration),
                                "evaluation_rows": len(evaluation),
                                "metrics": metrics(evaluation, scores, threshold),
                            }
                        )
        selected, summary, acceptable = select(results, policy["max_fpr"])
        name, target = selected.split("@")
        pooled = pd.concat(populations, ignore_index=True)
        train = pooled[pooled.day < policy["final_train_end_day"] - policy["feedback_delay_days"]]
        calibration = pooled[
            (pooled.day >= policy["final_train_end_day"])
            & (pooled.day < policy["final_train_end_day"] + policy["calibration_days"])
        ]
        names, kind = candidates[name]
        artifact, score, cs = fit(train, calibration, names, kind, policy)
        artifact["threshold"] = threshold_at_fpr(calibration.is_fraud, cs, float(target))
        artifact["model_version"] = (
            "robust-"
            + name
            + "-"
            + hashlib.sha256(
                json.dumps(artifact["parameters"], sort_keys=True).encode()
            ).hexdigest()[:12]
        )
        artifact["release"] = {
            "approved": False,
            "purpose": "synthetic demo; manual review required",
        }
        frozen = {
            "selected": selected,
            "policy": policy,
            "policy_sha256": digest(policy_path),
            "rolling_constraint_met": acceptable,
            "selection": "highest mean recall among candidates with every rolling FPR Wilson upper bound <= limit; otherwise lowest worst upper bound (unapproved)",
            "summary": summary,
            "threshold": artifact["threshold"],
            "populations": provenance,
        }
        model_path = output / "selected-model.json"
        model_path.write_text(json.dumps(artifact, allow_nan=False) + "\n")
        frozen["artifact_sha256"] = digest(model_path)
        (output / "selected-model.sha256").write_text(frozen["artifact_sha256"] + "\n")
        # Persist the exact decision before generating or scoring the unseen final population.
        (output / "selection-before-final.json").write_text(json.dumps(frozen, indent=2) + "\n")
        print(
            f"Frozen {selected}; generating untouched final population {policy['final_seed']}",
            flush=True,
        )
        final, final_config = population(policy, policy["final_seed"], "2025-01-01")
        # Evaluate the actual portable artifact, not just the training library predictor.
        portable = PortableModel(artifact)
        final_scores = np.array([portable.score(row) for row in final[names].to_dict("records")])
        native_scores = score(final)
        if not np.allclose(final_scores, native_scores, atol=1e-6, rtol=1e-5):
            raise ValueError("Final portable parity failed")
        final_metrics = metrics(final, final_scores, artifact["threshold"])
        rows = final[["transaction_id", "timestamp", "is_fraud", "fraud_scenario"]].copy()
        rows["score"] = final_scores
        rows["predicted"] = final_scores >= artifact["threshold"]
        rows.to_csv(output / "final-predictions.csv", index=False)
        report = {
            **frozen,
            "folds": results,
            "final": final_metrics,
            "final_population": {
                "config": final_config,
                "rows": len(final),
                "feature_sha256": hashlib.sha256(final.to_csv(index=False).encode()).hexdigest(),
            },
            "production_approved": False,
            "versions": {
                "python": platform.python_version(),
                "sklearn": sklearn.__version__,
                "torch": str(torch.__version__),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        }
        (output / "evaluation.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        (output / "REPORT.md").write_text(markdown(report))
    return report


def markdown(r):
    lines = [
        "# Rolling synthetic fraud evaluation",
        "",
        f"Selected: **{r['selected']}**. Rolling FPR constraint met: **{r['rolling_constraint_met']}**.",
        "",
        "Selection uses rolling development populations only. Final population was generated after freezing the candidate and threshold. No artifact is automatically approved or deployed.",
        "",
        "| Candidate / calibration FPR | Mean rolling recall | Worst rolling FPR | Worst FPR 95% upper | Eligible |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for key, m in r["summary"].items():
        lines.append(
            f"| {key} | {m['mean_recall']:.2%} | {m['max_fpr']:.2%} | {m['worst_fpr_95_upper']:.2%} | {m['fpr_constraint_met']} |"
        )
    m = r["final"]
    lines += [
        "",
        "## Untouched final population",
        "",
        f"Precision **{m['precision']:.2%}**, recall **{m['recall']:.2%}**, FPR **{m['false_positive_rate']:.2%}** ({m['false_positives_per_1000']:.2f} false positives per 1,000 legitimate transactions).",
        f"FPR Wilson 95% interval: {m['fpr_95_interval'][0]:.2%}–{m['fpr_95_interval'][1]:.2%}. Recall interval: {m['recall_95_interval'][0]:.2%}–{m['recall_95_interval'][1]:.2%}.",
        "",
        "| Fraud scenario | Cases | Detected | Recall |",
        "| --- | ---: | ---: | ---: |",
    ]
    for scenario, s in m["scenarios"].items():
        lines.append(f"| {scenario} | {s['fraud_cases']} | {s['detected']} | {s['recall']:.2%} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "- v2 has 18 behavioral features; v3 adds rolling spending and delayed confirmation features (28 total). Both MLPs use 32 hidden units, holding capacity fixed.",
        "- Targets 0.5%, 0.75%, 1% apply to calibration only. Selection requires the worst rolling FPR Wilson upper bound to be at most 1%; if none qualifies the least-exceeding candidate is reported without approval.",
        "- Seven-day feedback is revealed only at event time plus seven days. Training labels and calibration labels are subject to the same availability delay. Current hidden labels never enter current features.",
        "- Compromise groups rotate weekly. Seven-day feedback may be stale and can fail to improve terminal detection; the generator was not altered to make this task easier.",
        "- Wilson intervals describe binomial sampling uncertainty, not a guarantee under drift. Repeated customer/terminal observations are correlated; fold variation is also reported and intervals may be optimistic.",
        "- Synthetic populations do not establish real-world performance. Final metrics were not used to revise selection. Do not repeatedly tune against this final population.",
        "- Quality remains measured by scenario: 1 unusual purchases, 2 compromised terminals, 3 customer spending bursts. Scores are not calibrated fraud probabilities.",
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "fit",
    "markdown",
    "metrics",
    "population",
    "run",
    "select",
    "split_window",
    "validate_policy",
    "wilson",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=Path("configs/robust-evaluation.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.policy, args.output_dir)
    print(json.dumps({"selected": report["selected"], "final": report["final"]}, indent=2))
