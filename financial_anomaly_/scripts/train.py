"""Compatibility entry point for chronological training and release gates."""

from fraud_detector.model.evaluation import (
    average_precision,
    classification_metrics,
    precision_curve,
    select_threshold,
    split_by_time,
)
from fraud_detector.model.training import (
    as_tensor,
    export_artifact,
    fit_normalization,
    main,
    score_batches,
    train,
    train_model,
)

__all__ = [
    "as_tensor",
    "average_precision",
    "classification_metrics",
    "export_artifact",
    "fit_normalization",
    "main",
    "precision_curve",
    "score_batches",
    "select_threshold",
    "split_by_time",
    "train",
    "train_model",
]

if __name__ == "__main__":
    main()
