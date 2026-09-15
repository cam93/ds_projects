import os
from pathlib import Path

import torch

from fraud_detector.features.handbook import FEATURE_NAMES
from fraud_detector.schemas import Transaction


class FraudScorer:
    """Load the exported classifier and apply its training-time normalization."""

    def __init__(self, artifact_path: str | Path | None = None) -> None:
        self.artifact_path = Path(
            artifact_path or os.getenv("MODEL_PATH", "models/artifacts/fraud_model.pt")
        )
        if not self.artifact_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {self.artifact_path}. Run scripts/train.py first."
            )
        artifact = torch.load(self.artifact_path, map_location="cpu", weights_only=True)
        if artifact["feature_names"] != FEATURE_NAMES:
            raise ValueError("Model feature order does not match the feature handbook")
        self.means = artifact["normalization_means"]
        self.stds = artifact["normalization_stds"]
        self.threshold = float(artifact["threshold"])
        self.model_version = str(artifact["model_version"])
        self.model = torch.nn.Sequential(
            torch.nn.Linear(len(FEATURE_NAMES), 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1),
        )
        self.model.load_state_dict(artifact["model_state_dict"])
        self.model.eval()

    def _features(self, transaction: Transaction) -> torch.Tensor:
        values = {
            "amount": transaction.amount,
            "transactions_last_hour": transaction.transactions_last_hour,
            "customer_history_days": transaction.customer_history_days,
            "hour_of_day": transaction.hour_of_day,
        }
        return torch.tensor(
            [[(values[name] - self.means[name]) / self.stds[name] for name in FEATURE_NAMES]],
            dtype=torch.float32,
        )

    def predict(self, transaction: Transaction) -> tuple[float, bool]:
        with torch.inference_mode():
            probability = float(torch.sigmoid(self.model(self._features(transaction))).item())
        return probability, probability >= self.threshold
