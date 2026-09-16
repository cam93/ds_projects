import hashlib
import io
import json
import math
import os
from pathlib import Path

import torch

from fraud_detector.features.adaptive import V3_FEATURE_NAMES
from fraud_detector.features.handbook import FEATURE_NAMES, V2_FEATURE_NAMES
from fraud_detector.model.portable import PortableModel
from fraud_detector.schemas import Transaction


class MissingBehavioralFeatures(ValueError):
    pass


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
        contents = self.artifact_path.read_bytes()
        expected = os.getenv("MODEL_SHA256")
        if os.getenv("APP_ENV", "production") == "production" and not expected:
            raise ValueError("MODEL_SHA256 must pin the approved artifact in production")
        if expected and hashlib.sha256(contents).hexdigest() != expected:
            raise ValueError("Model artifact SHA256 mismatch")
        is_portable = self.artifact_path.suffix == ".json"
        artifact = (
            json.loads(contents)
            if is_portable
            else torch.load(io.BytesIO(contents), map_location="cpu", weights_only=True)
        )
        self.feature_names = artifact["feature_names"]
        if self.feature_names not in (FEATURE_NAMES, V2_FEATURE_NAMES, V3_FEATURE_NAMES):
            raise ValueError("Model feature order does not match the feature handbook")
        if os.getenv("APP_ENV", "production") == "production" and not artifact.get(
            "release", {}
        ).get("approved"):
            raise ValueError("Model has not passed release evaluation gates")
        self.feedback_delay_days = artifact.get("feedback_delay_days", 7)
        if self.feature_names == V3_FEATURE_NAMES and (
            not is_portable
            or not isinstance(self.feedback_delay_days, (int, float))
            or not 1 <= self.feedback_delay_days <= 30
        ):
            raise ValueError("v3 requires portable JSON with a valid feedback delay")
        self.threshold = float(artifact["threshold"])
        self.model_version = str(artifact["model_version"])
        if not math.isfinite(self.threshold) or not 0 <= self.threshold <= (
            2 if is_portable else 1
        ):
            raise ValueError("Invalid model threshold")
        self.portable = PortableModel(artifact) if is_portable else None
        if self.portable is not None:
            return
        self.means = artifact["normalization_means"]
        self.stds = artifact["normalization_stds"]
        for name in self.feature_names:
            if (
                not math.isfinite(self.means[name])
                or not math.isfinite(self.stds[name])
                or self.stds[name] <= 0
            ):
                raise ValueError("Invalid normalization parameters")
        self.model = torch.nn.Sequential(
            torch.nn.Linear(len(self.feature_names), 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1),
        )
        self.model.load_state_dict(artifact["model_state_dict"])
        if any(not torch.isfinite(value).all() for value in self.model.state_dict().values()):
            raise ValueError("Nonfinite model weights")
        self.model.eval()

    def _features(self, transaction: Transaction) -> torch.Tensor:
        values = {
            "amount": transaction.amount,
            "transactions_last_hour": transaction.transactions_last_hour,
            "customer_history_days": transaction.customer_history_days,
            "hour_of_day": transaction.hour_of_day,
        }
        if transaction.behavioral_features is not None:
            values.update(transaction.behavioral_features.model_dump())
        return torch.tensor(
            [[(values[name] - self.means[name]) / self.stds[name] for name in self.feature_names]],
            dtype=torch.float32,
        )

    def predict(self, transaction: Transaction) -> tuple[float, bool]:
        if (
            self.feature_names in (V2_FEATURE_NAMES, V3_FEATURE_NAMES)
            and transaction.behavioral_features is None
        ):
            raise MissingBehavioralFeatures(
                "This model requires the complete behavioral_features block"
            )
        if self.feature_names == V3_FEATURE_NAMES and (
            transaction.adaptive_features is None
            or transaction.adaptive_features.feedback_delay_days != self.feedback_delay_days
        ):
            raise MissingBehavioralFeatures(
                "This model requires adaptive_features with its trained feedback delay"
            )
        if self.portable is not None:
            values = transaction.model_dump(exclude={"behavioral_features"})
            if transaction.behavioral_features is not None:
                values.update(transaction.behavioral_features.model_dump())
            if transaction.adaptive_features is not None:
                values.update(transaction.adaptive_features.model_dump())
            probability = self.portable.score(values)
            return probability, probability >= self.threshold
        with torch.inference_mode():
            probability = float(torch.sigmoid(self.model(self._features(transaction))).item())
        if not math.isfinite(probability):
            raise ValueError("Nonfinite prediction")
        return probability, probability >= self.threshold
