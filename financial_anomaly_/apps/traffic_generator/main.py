"""Bounded, restartable replay of trusted prepared events."""

import hashlib
import json
import logging
import math
import os
import time
from pathlib import Path
from uuid import uuid4

import httpx

from fraud_detector.delivery import post_transaction
from fraud_detector.schemas import Transaction
from fraud_detector.security import secret

logger = logging.getLogger(__name__)


def replay_transactions(data_path: Path, replay_run_id: str):
    with data_path.open(encoding="utf-8") as replay_file:
        for number, line in enumerate(replay_file, 1):
            if not line.strip():
                continue
            if len(line) > 16384:
                raise ValueError(f"Replay line {number} too large")
            event = json.loads(line)
            source_id = str(event["transaction_id"])
            event["source_transaction_id"] = source_id
            event["transaction_id"] = f"{replay_run_id}:{source_id}"
            yield Transaction.model_validate(event).model_dump(mode="json", exclude_none=True)


def atomic_checkpoint(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as output:
        json.dump(state, output)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_replay(data_path, checkpoint_path, client, interval=0.25, run_id=None):
    if interval < 0 or not math.isfinite(interval):
        raise ValueError("Replay interval must be finite and nonnegative")
    digest = file_digest(data_path)
    if checkpoint_path.exists():
        state = json.loads(checkpoint_path.read_text())
        if state["dataset_sha256"] != digest or (run_id and run_id != state["run_id"]):
            raise ValueError("Checkpoint does not match dataset/run; use a new checkpoint path")
    else:
        state = {"dataset_sha256": digest, "run_id": run_id or uuid4().hex[:16], "completed": 0}
        atomic_checkpoint(checkpoint_path, state)
    for number, event in enumerate(replay_transactions(data_path, state["run_id"]), 1):
        if number <= state["completed"]:
            continue
        post_transaction(client, event)
        state["completed"] = number
        atomic_checkpoint(checkpoint_path, state)
        if number % 100 == 0:
            logger.info("replay_progress completed=%d", number)
        time.sleep(interval)
    logger.info("replay_complete completed=%d", state["completed"])
    return state


def main():
    logging.basicConfig(level=logging.INFO)
    with httpx.Client(
        base_url=os.getenv("TRAFFIC_TARGET_URL", "http://api:8000"),
        headers={"Authorization": f"Bearer {secret('API_KEY')}"},
        timeout=5,
        trust_env=False,
    ) as client:
        run_replay(
            Path(os.getenv("TRAFFIC_DATA_PATH", "/data/replay.jsonl")),
            Path(os.getenv("TRAFFIC_CHECKPOINT_PATH", "/state/checkpoint.json")),
            client,
            float(os.getenv("TRAFFIC_INTERVAL_SECONDS", ".25")),
            os.getenv("TRAFFIC_REPLAY_RUN_ID"),
        )


__all__ = ["atomic_checkpoint", "main", "post_transaction", "replay_transactions", "run_replay"]


if __name__ == "__main__":
    main()
