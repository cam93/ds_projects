"""Generate continuous synthetic transaction traffic for local demonstrations."""

import json
import os
import random
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def build_transaction() -> dict[str, object]:
    return {
        "transaction_id": str(uuid4()),
        "customer_id": str(random.randint(1, 5000)),
        "terminal_id": str(random.randint(1, 10000)),
        "amount": round(random.uniform(2.0, 2500.0), 2),
        "transactions_last_hour": random.randint(0, 20),
        "customer_history_days": random.randint(1, 3650),
        "hour_of_day": datetime.now(timezone.utc).hour,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def replay_transactions(data_path: Path, replay_run_id: str) -> list[dict[str, object]]:
    with data_path.open(encoding="utf-8") as replay_file:
        events = []
        for line in replay_file:
            event = json.loads(line)
            source_id = str(event["transaction_id"])
            event["source_transaction_id"] = source_id
            event["transaction_id"] = f"{replay_run_id}:{source_id}"
            events.append(event)
        return events


def post_transaction(endpoint: str, transaction: dict[str, object]) -> None:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(transaction).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5):
        pass


def main() -> None:
    target_url = os.getenv("TRAFFIC_TARGET_URL", "http://localhost:8000")
    interval = float(os.getenv("TRAFFIC_INTERVAL_SECONDS", "0.25"))
    endpoint = f"{target_url.rstrip('/')}/predict"
    source = os.getenv("TRAFFIC_SOURCE", "random").lower()
    if source == "handbook":
        data_path = Path(os.getenv("TRAFFIC_DATA_PATH", "/data/replay.jsonl"))
        replay_run_id = os.getenv("TRAFFIC_REPLAY_RUN_ID", f"run-{uuid4().hex[:8]}")
        transactions = replay_transactions(data_path, replay_run_id)
        for transaction in transactions:
            while True:
                try:
                    post_transaction(endpoint, transaction)
                    break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(1)
            time.sleep(interval)
        return
    if source != "random":
        raise ValueError(f"Unsupported TRAFFIC_SOURCE: {source}")
    while True:
        try:
            post_transaction(endpoint, build_transaction())
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1)
        time.sleep(interval)


if __name__ == "__main__":
    main()
