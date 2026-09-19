"""Measure sequential requests through the local demo API, including ledger persistence."""

import argparse
import json
import time
from pathlib import Path
from uuid import uuid4

import httpx
import numpy as np

from fraud_detector.features.adaptive import AdaptiveState
from fraud_detector.features.handbook import FeatureState
from fraud_detector.features.pipeline import enrich_transaction, prepared_transaction
from fraud_detector.simulation import SimulationConfig, TransactionWorld, parse_start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--key-file", type=Path, default=Path("secrets/api_key"))
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.count <= 1000:
        parser.error("count must be in 1..1000")
    if args.output.exists():
        parser.error("Use a new report filename")
    artifact = json.loads(Path("models/artifacts/fraud_model.json").read_text())
    world = TransactionWorld(SimulationConfig(seed=2026), parse_start("2025-12-01"))
    history, adaptive = FeatureState(), AdaptiveState()
    run_id = "benchmark-" + uuid4().hex[:12]
    latencies, failures, versions = [], [], set()
    with httpx.Client(
        base_url=args.url,
        headers={"Authorization": "Bearer " + args.key_file.read_text().strip()},
        timeout=10,
        trust_env=False,
    ) as client:
        client.get("/ready").raise_for_status()
        started = time.perf_counter()
        for raw in world.day(0)[: args.count]:
            features = enrich_transaction(raw, history, adaptive)
            event = prepared_transaction(
                transaction_id=run_id + ":" + str(raw["TRANSACTION_ID"]),
                customer_id=raw["CUSTOMER_ID"],
                terminal_id=raw["TERMINAL_ID"],
                timestamp=raw["TX_DATETIME"],
                features=features,
            )
            before = time.perf_counter()
            response = client.post(
                "/predict", json=event.model_dump(mode="json", exclude_none=True)
            )
            elapsed = (time.perf_counter() - before) * 1000
            if response.status_code != 200:
                failures.append(response.status_code)
                continue
            versions.add(response.json()["model_version"])
            latencies.append(elapsed)
        duration = time.perf_counter() - started
    report = {
        "model_versions": sorted(versions),
        "expected_model_version": artifact["model_version"],
        "successes": len(latencies),
        "failures": failures,
        "elapsed_seconds": duration,
        "successful_requests_per_second": len(latencies) / duration,
        "p50_ms": float(np.percentile(latencies, 50)) if latencies else None,
        "p95_ms": float(np.percentile(latencies, 95)) if latencies else None,
        "p99_ms": float(np.percentile(latencies, 99)) if latencies else None,
        "method": "Sequential localhost HTTP requests including audit persistence; simulator may run concurrently. Not a saturation or HA load test.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if failures or versions != {artifact["model_version"]} or len(latencies) != args.count:
        raise SystemExit("API benchmark failed or used the wrong model")


if __name__ == "__main__":
    main()
