"""Train, evaluate, and export the fraud classifier."""

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn

from fraud_detector.features.handbook import FEATURE_NAMES

MODEL_VERSION = "fraud-mlp-v1"
SPLITS = (0.70, 0.15, 0.15)


def split_by_time(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = records.sort_values(["timestamp", "transaction_id"], kind="mergesort")
    train_end = int(len(ordered) * SPLITS[0])
    validation_end = train_end + int(len(ordered) * SPLITS[1])
    return ordered.iloc[:train_end], ordered.iloc[train_end:validation_end], ordered.iloc[validation_end:]


def fit_normalization(training: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    means = {name: float(training[name].mean()) for name in FEATURE_NAMES}
    stds = {
        name: float(training[name].std(ddof=0)) or 1.0
        for name in FEATURE_NAMES
    }
    return means, stds


def as_tensor(
    records: pd.DataFrame,
    means: dict[str, float],
    stds: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    values = [[(float(row[name]) - means[name]) / stds[name] for name in FEATURE_NAMES]
              for _, row in records.iterrows()]
    labels = records["is_fraud"].astype("float32").to_numpy()
    return torch.tensor(values, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


def average_precision(labels: list[int], scores: list[float]) -> float:
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    positives = sum(labels)
    if positives == 0:
        return 0.0
    found = 0
    precision_sum = 0.0
    for rank, index in enumerate(order, start=1):
        if labels[index]:
            found += 1
            precision_sum += found / rank
    return precision_sum / positives


def classification_metrics(labels: list[int], scores: list[float], threshold: float) -> dict[str, float]:
    predictions = [score >= threshold for score in scores]
    true_positive = sum(prediction and label for prediction, label in zip(predictions, labels))
    false_positive = sum(prediction and not label for prediction, label in zip(predictions, labels))
    false_negative = sum(not prediction and label for prediction, label in zip(predictions, labels))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "average_precision": average_precision(labels, scores),
    }


def select_threshold(labels: list[int], scores: list[float]) -> float:
    if not labels or sum(labels) == 0 or sum(labels) == len(labels):
        return 0.5
    candidates = [index / 100 for index in range(5, 100)]
    return max(
        candidates,
        key=lambda threshold: (
            2
            * classification_metrics(labels, scores, threshold)["precision"]
            * classification_metrics(labels, scores, threshold)["recall"]
            / (
                classification_metrics(labels, scores, threshold)["precision"]
                + classification_metrics(labels, scores, threshold)["recall"]
            )
            if classification_metrics(labels, scores, threshold)["precision"]
            + classification_metrics(labels, scores, threshold)["recall"]
            else 0.0,
            threshold,
        ),
    )


def train_model(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    means: dict[str, float],
    stds: dict[str, float],
    epochs: int,
) -> tuple[nn.Module, float, dict[str, float]]:
    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(len(FEATURE_NAMES), 8), nn.ReLU(), nn.Linear(8, 1))
    train_x, train_y = as_tensor(training, means, stds)
    validation_x, validation_y = as_tensor(validation, means, stds)
    positives = train_y.sum()
    negatives = len(train_y) - positives
    loss_function = nn.BCEWithLogitsLoss(pos_weight=negatives / positives if positives else 1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss_function(model(train_x).squeeze(1), train_y).backward()
        optimizer.step()
    with torch.inference_mode():
        validation_scores = torch.sigmoid(model(validation_x).squeeze(1)).tolist()
    labels = validation_y.to(torch.int64).tolist()
    threshold = select_threshold(labels, validation_scores)
    return model, threshold, classification_metrics(labels, validation_scores, threshold)


def export_artifact(
    model: nn.Module,
    means: dict[str, float],
    stds: dict[str, float],
    threshold: float,
    metrics: dict[str, float],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_names": FEATURE_NAMES,
            "normalization_means": means,
            "normalization_stds": stds,
            "threshold": threshold,
            "model_version": MODEL_VERSION,
            "metrics": metrics,
        },
        output_path,
    )


def train(input_path: Path, output_path: Path, epochs: int = 50) -> dict[str, Any]:
    records = pd.read_csv(input_path)
    train_records, validation_records, test_records = split_by_time(records)
    means, stds = fit_normalization(train_records)
    model, threshold, validation_metrics = train_model(
        train_records, validation_records, means, stds, epochs
    )
    test_x, test_y = as_tensor(test_records, means, stds)
    with torch.inference_mode():
        test_scores = torch.sigmoid(model(test_x).squeeze(1)).tolist()
    test_labels = test_y.to(torch.int64).tolist()
    test_metrics = classification_metrics(test_labels, test_scores, threshold)
    export_artifact(model, means, stds, threshold, test_metrics, output_path)
    report = {
        "model_version": MODEL_VERSION,
        "threshold": threshold,
        "validation": validation_metrics,
        "test": test_metrics,
        "rows": {"train": len(train_records), "validation": len(validation_records), "test": len(test_records)},
    }
    output_path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/training_features.csv"))
    parser.add_argument("--output", type=Path, default=Path("models/artifacts/fraud_model.pt"))
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    random.seed(7)
    report = train(args.input, args.output, args.epochs)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
