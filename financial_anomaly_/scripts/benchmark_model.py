"""Evaluate the active portable model on a fresh population; never tune or train it."""

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from fraud_detector.artifacts import digest
from fraud_detector.dataset import population
from fraud_detector.features.pipeline import prepared_transaction
from fraud_detector.model.evaluation import metrics
from fraud_detector.model.inference import FraudScorer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/artifacts/fraud_model.json"))
    parser.add_argument("--seed", type=int, default=1909)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    # Explicit offline synthetic benchmark; production gates remain enforced by the API.
    os.environ["APP_ENV"] = "development"
    os.environ["MODEL_SHA256"] = digest(args.model)
    model = FraudScorer(args.model)
    policy = json.loads(Path("configs/robust-evaluation.json").read_text())
    if args.seed in policy["training_seeds"] + [policy["final_seed"]]:
        raise ValueError("Use a seed outside the development and previously evaluated populations")
    frame, config = population(policy, args.seed, "2025-07-01")
    scores, latencies = [], []
    started = time.perf_counter()
    for row in frame.itertuples(index=False):
        event = prepared_transaction(
            transaction_id=row.transaction_id,
            customer_id=row.customer_id,
            terminal_id=row.terminal_id,
            timestamp=row.timestamp,
            features=row._asdict(),
            feedback_delay_days=policy["feedback_delay_days"],
        )
        before = time.perf_counter()
        score, _ = model.predict(event)
        latencies.append((time.perf_counter() - before) * 1000)
        scores.append(score)
    elapsed = time.perf_counter() - started
    result = metrics(frame, np.asarray(scores), model.threshold)
    result["accuracy"] = (result["true_positive"] + result["true_negative"]) / len(frame)
    report = {
        "model_version": model.model_version,
        "model_sha256": digest(args.model),
        "feature_count": len(model.feature_names),
        "threshold": model.threshold,
        "rows": len(frame),
        "population": config,
        "synthetic": True,
        "metrics": result,
        "local_scoring": {
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "transactions_per_second_including_validation": len(frame) / elapsed,
            "elapsed_seconds": elapsed,
            "platform": platform.platform(),
        },
        "limitations": "Local sequential scoring plus schema validation; excludes HTTP, ledger and network overhead. No tuning, retraining or artifact modification.",
    }
    (args.output_dir / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
