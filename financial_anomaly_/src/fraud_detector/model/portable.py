"""Data-only JSON scoring for benchmark candidates; never unpickle sklearn objects."""

import math

from fraud_detector.features.handbook import FEATURE_NAMES, V2_FEATURE_NAMES


def sigmoid(value):
    return 1 / (1 + math.exp(-value)) if value >= 0 else math.exp(value) / (1 + math.exp(value))


class PortableModel:
    def __init__(self, artifact):
        self.artifact = artifact
        self.names = artifact["feature_names"]
        if artifact.get("format_version") != 1 or self.names not in (
            FEATURE_NAMES,
            V2_FEATURE_NAMES,
        ):
            raise ValueError("Unsupported portable feature schema")
        self.kind = artifact["model_type"]
        self.means, self.scales = artifact["means"], artifact["scales"]
        size = len(self.names)

        def vector(values, length):
            if len(values) != length or not all(
                isinstance(v, (int, float)) and math.isfinite(v) for v in values
            ):
                raise ValueError("Invalid portable model vector")

        vector(self.means, size)
        vector(self.scales, size)
        if any(v <= 0 for v in self.scales):
            raise ValueError("Invalid portable normalization scale")
        self.parameters = artifact["parameters"]
        p = self.parameters
        if self.kind == "logistic":
            vector(p["coefficients"], size)
            vector([p["intercept"]], 1)
        elif self.kind == "mlp":
            width = len(p["hidden_bias"])
            if not 1 <= width <= 256 or len(p["hidden_weights"]) != width:
                raise ValueError("Invalid portable MLP dimensions")
            for row in p["hidden_weights"]:
                vector(row, size)
            vector(p["hidden_bias"], width)
            vector(p["output_weights"], width)
            vector([p["output_bias"]], 1)
        elif self.kind == "hist_gradient_boosting":
            vector([p["baseline"]], 1)
            if not 1 <= len(p["trees"]) <= 1000:
                raise ValueError("Invalid portable tree count")
            for nodes in p["trees"]:
                if not 1 <= len(nodes) <= 2047:
                    raise ValueError("Invalid portable tree size")
                for index, node in enumerate(nodes):
                    if "value" in node:
                        vector([node["value"]], 1)
                    elif (
                        not isinstance(node["feature"], int)
                        or not 0 <= node["feature"] < size
                        or not isinstance(node["left"], int)
                        or not isinstance(node["right"], int)
                        or not index < node["left"] < len(nodes)
                        or not index < node["right"] < len(nodes)
                    ):
                        raise ValueError("Invalid or cyclic portable tree")
                    else:
                        vector([node["threshold"]], 1)
        else:
            raise ValueError("Unsupported portable model type")

    def score(self, values):
        if any(name not in values or not math.isfinite(values[name]) for name in self.names):
            raise ValueError("Model requires its complete finite feature set")
        x = [
            (values[name] - mean) / scale
            for name, mean, scale in zip(self.names, self.means, self.scales)
        ]
        p = self.parameters
        if self.kind == "logistic":
            raw = p["intercept"] + sum(
                weight * value for weight, value in zip(p["coefficients"], x)
            )
        elif self.kind == "mlp":
            hidden = [
                max(0.0, bias + sum(weight * value for weight, value in zip(weights, x)))
                for weights, bias in zip(p["hidden_weights"], p["hidden_bias"])
            ]
            raw = p["output_bias"] + sum(
                weight * value for weight, value in zip(p["output_weights"], hidden)
            )
        else:
            raw = p["baseline"]
            for nodes in p["trees"]:
                index = 0
                while "value" not in nodes[index]:
                    node = nodes[index]
                    index = (
                        node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
                    )
                raw += nodes[index]["value"]
        if not math.isfinite(raw):
            raise ValueError("Nonfinite model prediction")
        return sigmoid(raw)
