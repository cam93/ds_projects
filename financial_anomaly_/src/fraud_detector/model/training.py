"""Chronological, minibatch training with explicit release gates."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fraud_detector.features.handbook import FEATURE_NAMES
from fraud_detector.model.evaluation import classification_metrics, select_threshold, split_by_time


def fit_normalization(training, feature_names=None):
    feature_names = feature_names or FEATURE_NAMES
    values = training[feature_names].to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Training features must be non-empty and finite")
    stds = values.std(axis=0)
    return (
        dict(zip(feature_names, values.mean(axis=0).tolist())),
        dict(zip(feature_names, np.where(stds > 0, stds, 1).tolist())),
    )


def as_tensor(records, means, stds, feature_names=None):
    feature_names = feature_names or FEATURE_NAMES
    values = records[feature_names].to_numpy(dtype=np.float32)
    values = (values - np.array([means[n] for n in feature_names], dtype=np.float32)) / np.array(
        [stds[n] for n in feature_names], dtype=np.float32
    )
    return torch.from_numpy(values), torch.tensor(records.is_fraud.to_numpy(dtype=np.float32))


def fit_mlp(training, means, stds, epochs, feature_names=None, seed=7, hidden_size=8):
    feature_names = feature_names or FEATURE_NAMES
    if epochs < 1:
        raise ValueError("epochs must be positive")
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(len(feature_names), hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
    )
    x, y = as_tensor(training, means, stds, feature_names)
    if not 0 < y.sum() < len(y):
        raise ValueError("Training requires both legitimate and fraudulent examples")
    loss = nn.BCEWithLogitsLoss(pos_weight=(len(y) - y.sum()) / y.sum())
    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=1024,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss(model(batch_x).squeeze(1), batch_y).backward()
            optimizer.step()
    model.eval()
    return model


def train_model(
    training, validation, means, stds, epochs, feature_names=None, seed=7, hidden_size=8
):
    model = fit_mlp(training, means, stds, epochs, feature_names, seed, hidden_size)
    validation_x, validation_y = as_tensor(validation, means, stds, feature_names)
    scores = score_batches(model, validation_x)
    labels = validation_y.to(torch.int64).tolist()
    threshold = select_threshold(labels, scores)
    return model, threshold, classification_metrics(labels, scores, threshold)


def score_batches(model, features):
    with torch.inference_mode():
        return torch.cat(
            [torch.sigmoid(model(batch).squeeze(1)) for batch in features.split(4096)]
        ).tolist()


def export_artifact(
    model, means, stds, threshold, metrics, output_path, *, release=None, version="test-model"
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_names": FEATURE_NAMES,
            "normalization_means": means,
            "normalization_stds": stds,
            "threshold": threshold,
            "model_version": version,
            "metrics": metrics,
            "release": release or {"approved": False},
        },
        temporary,
    )
    os.replace(temporary, output_path)


def train(input_path, output_path, epochs=50, policy_path=None):
    if Path(output_path).exists():
        raise ValueError("Use a new candidate output path; existing model artifacts are immutable")
    if policy_path is None:
        raise ValueError("A reviewed release policy is required (--policy)")
    policy = json.loads(Path(policy_path).read_text())
    required = {
        "min_positive_per_split",
        "min_negative_per_split",
        "min_precision",
        "min_recall",
        "min_average_precision",
    }
    if not required <= policy.keys():
        raise ValueError("Release policy is incomplete")
    for key in ("min_positive_per_split", "min_negative_per_split"):
        if not isinstance(policy[key], int) or policy[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("min_precision", "min_recall", "min_average_precision"):
        if not 0 < policy[key] <= 1:
            raise ValueError(f"{key} must be in (0,1]")
    records = pd.read_csv(input_path, dtype={"transaction_id": str})
    if records.transaction_id.isna().any() or records.transaction_id.duplicated().any():
        raise ValueError("Transaction IDs must be present and unique")
    if (
        not records.is_fraud.isin([0, 1]).all()
        or not np.isfinite(records[FEATURE_NAMES].to_numpy(dtype=float)).all()
    ):
        raise ValueError("Invalid labels or features")
    splits = split_by_time(records)
    counts = {}
    for name, split in zip(("train", "validation", "test"), splits):
        positive = int(split.is_fraud.sum())
        negative = len(split) - positive
        counts[name] = {"rows": len(split), "positive": positive, "negative": negative}
        if (
            positive < policy["min_positive_per_split"]
            or negative < policy["min_negative_per_split"]
        ):
            raise ValueError(f"Insufficient evaluation data in {name}: {counts[name]}")
    training, validation, test = splits
    means, stds = fit_normalization(training)
    model, threshold, validation_metrics = train_model(training, validation, means, stds, epochs)
    x, y = as_tensor(test, means, stds)
    metrics = classification_metrics(y.int().tolist(), score_batches(model, x), threshold)
    approved = all(
        metrics[name] >= policy[f"min_{name}"]
        for name in ("precision", "recall", "average_precision")
    )
    data_hash = hashlib.sha256(Path(input_path).read_bytes()).hexdigest()
    policy_hash = hashlib.sha256(Path(policy_path).read_bytes()).hexdigest()
    weights_hash = hashlib.sha256(
        b"".join(t.detach().numpy().tobytes() for t in model.state_dict().values())
    ).hexdigest()
    version = f"mlp-{data_hash[:12]}-{weights_hash[:12]}"
    report = {
        "approved": approved,
        "model_version": version,
        "threshold": threshold,
        "validation": validation_metrics,
        "test": metrics,
        "splits": counts,
        "dataset_sha256": data_hash,
        "policy_sha256": policy_hash,
        "policy": policy,
        "torch_version": str(torch.__version__),
        "seed": 7,
        "epochs": epochs,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    if not approved:
        raise ValueError("Release gate failed; report saved, model not promoted")
    export_artifact(
        model, means, stds, threshold, metrics, output_path, release=report, version=version
    )
    output_path.with_suffix(".sha256").write_text(
        hashlib.sha256(output_path.read_bytes()).hexdigest() + "\n"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/training_features.csv"))
    parser.add_argument("--output", type=Path, default=Path("models/artifacts/fraud_model.pt"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--policy", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(train(args.input, args.output, args.epochs, args.policy), indent=2))


if __name__ == "__main__":
    main()
