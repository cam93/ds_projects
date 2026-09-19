"""One fitting/scoring implementation for both comparison workflows.

Selection and release decisions belong to the caller. This module only fits the fixed
candidate configurations and checks that portable export preserves their predictions.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from fraud_detector.model.export import export_parameters
from fraud_detector.model.portable import PortableModel
from fraud_detector.model.training import as_tensor, fit_mlp, fit_normalization, score_batches


@dataclass
class FittedCandidate:
    model: Any
    names: list[str]
    kind: str
    means: dict[str, float]
    stds: dict[str, float]
    hyperparameters: dict

    @property
    def mean_vector(self):
        return [self.means[name] for name in self.names]

    @property
    def scale_vector(self):
        return [self.stds[name] for name in self.names]

    @property
    def parameters(self):
        return export_parameters(self.kind, self.model)

    def score(self, frame):
        if self.kind == "mlp":
            features, _ = as_tensor(frame, self.means, self.stds, self.names)
            return np.asarray(score_batches(self.model, features))
        values = (frame[self.names].to_numpy(dtype=float) - self.mean_vector) / self.scale_vector
        return self.model.predict_proba(values)[:, 1]

    def verify_export(self, frame, scores, artifact):
        indices = np.linspace(0, len(frame) - 1, min(1000, len(frame)), dtype=int)
        portable = PortableModel(artifact)
        exported = np.array(
            [portable.score(row) for row in frame.iloc[indices][self.names].to_dict("records")]
        )
        expected = np.asarray(scores)[indices]
        if not np.allclose(exported, expected, atol=1e-6, rtol=1e-5):
            raise ValueError("Portable/native scoring mismatch")
        return float(np.max(np.abs(exported - expected)))


def fit_candidate(training, names, kind, *, epochs, seed, hidden_size):
    means, stds = fit_normalization(training, names)
    if kind == "hist_gradient_boosting":
        means, stds = dict.fromkeys(names, 0.0), dict.fromkeys(names, 1.0)
    if kind == "mlp":
        model = fit_mlp(training, means, stds, epochs, names, seed, hidden_size)
        parameters = {
            "epochs": epochs,
            "hidden_size": hidden_size,
            "batch_size": 1024,
            "learning_rate": 0.001,
            "positive_class_weight": "balanced",
        }
    else:
        if kind == "logistic":
            model = LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=2000, random_state=seed
            )
        elif kind == "hist_gradient_boosting":
            model = HistGradientBoostingClassifier(
                max_iter=150,
                max_leaf_nodes=15,
                min_samples_leaf=50,
                learning_rate=0.08,
                l2_regularization=1.0,
                class_weight="balanced",
                early_stopping=False,
                random_state=seed,
            )
        else:
            raise ValueError(f"Unsupported candidate type: {kind}")
        values = (training[names].to_numpy(dtype=float) - [means[n] for n in names]) / [
            stds[n] for n in names
        ]
        model.fit(values, training.is_fraud)
        parameters = model.get_params()
    return FittedCandidate(model, names, kind, means, stds, parameters)
