"""Shared chronological splits and evaluation metrics; no training or CLI side effects."""

import numpy as np
import pandas as pd


def split_by_time(records):
    ordered = records.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], utc=True, errors="raise")
    ordered = ordered.sort_values(["timestamp", "transaction_id"], kind="mergesort")
    times = ordered["timestamp"].drop_duplicates().sort_values().tolist()
    if len(times) < 7:
        raise ValueError("At least seven distinct timestamps are required")
    train_cut = times[int(len(times) * 0.70)]
    test_cut = times[int(len(times) * 0.85)]
    return (
        ordered[ordered.timestamp < train_cut],
        ordered[(ordered.timestamp >= train_cut) & (ordered.timestamp < test_cut)],
        ordered[ordered.timestamp >= test_cut],
    )


def precision_curve(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if not len(labels) or len(labels) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Scores and labels must be finite, non-empty, and aligned")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be binary")
    order = np.argsort(-scores, kind="stable")
    y, s = labels[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp = np.cumsum(y)[ends]
    precision = tp / (ends + 1)
    recall = tp / max(1, labels.sum())
    return s[ends], precision, recall


def average_precision(labels, scores):
    _, precision, recall = precision_curve(labels, scores)
    return float(np.sum(np.diff(np.r_[0, recall]) * precision))


def classification_metrics(labels, scores, threshold):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predictions = scores >= threshold
    tp = int(np.sum(predictions & (labels == 1)))
    fp = int(np.sum(predictions & (labels == 0)))
    fn = int(np.sum(~predictions & (labels == 1)))
    tn = int(np.sum(~predictions & (labels == 0)))
    return {
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "average_precision": average_precision(labels, scores),
        "false_positive_rate": fp / max(1, fp + tn),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
    }


def select_threshold(labels, scores):
    if not 0 < sum(labels) < len(labels):
        raise ValueError("Threshold selection requires both classes")
    thresholds, precision, recall = precision_curve(labels, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[int(np.argmax(f1))])


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
    from sklearn.metrics import roc_auc_score

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


def wilson(successes, total, z=1.96):
    if not total:
        return [None, None]
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, float(center - margin)), min(1.0, float(center + margin))]


def metrics(frame, scores, threshold):
    result = assess(frame, scores, threshold)
    y = frame.is_fraud.to_numpy(dtype=int)
    predicted = np.asarray(scores) >= threshold
    fp, tp = int((predicted & (y == 0)).sum()), int((predicted & (y == 1)).sum())
    result["fpr_95_interval"] = wilson(fp, int((y == 0).sum()))
    result["recall_95_interval"] = wilson(tp, int(y.sum()))
    result["precision_95_interval"] = wilson(tp, int(predicted.sum()))
    result["false_positives_per_1000"] = result["false_positive_rate"] * 1000
    return result
