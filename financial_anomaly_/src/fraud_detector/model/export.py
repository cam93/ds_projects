"""Data-only parameter export for supported fitted classifiers."""


def export_parameters(kind, model):
    if kind == "logistic":
        return {"coefficients": model.coef_[0].tolist(), "intercept": float(model.intercept_[0])}
    if kind == "mlp":
        return {
            "hidden_weights": model[0].weight.detach().tolist(),
            "hidden_bias": model[0].bias.detach().tolist(),
            "output_weights": model[2].weight.detach()[0].tolist(),
            "output_bias": float(model[2].bias.detach()[0]),
        }
    trees = []
    # Private sklearn representation is pinned and verified against native predict_proba.
    # Only numeric finite features are allowed: no categorical/missing-value routing.
    for predictors in model._predictors:
        if len(predictors) != 1:
            raise ValueError("Only binary numeric boosting models are supported")
        nodes = []
        for node in predictors[0].nodes:
            if node["is_leaf"]:
                nodes.append({"value": float(node["value"])})
            else:
                if node["is_categorical"]:
                    raise ValueError("Categorical trees are unsupported")
                nodes.append(
                    {
                        "feature": int(node["feature_idx"]),
                        "threshold": float(node["num_threshold"]),
                        "left": int(node["left"]),
                        "right": int(node["right"]),
                    }
                )
        trees.append(nodes)
    return {"baseline": float(model._baseline_prediction[0, 0]), "trees": trees}
