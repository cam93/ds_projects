"""Batch generation and checkpointed live delivery of synthetic Handbook transactions."""

import argparse
import csv
import fcntl
import hashlib
import json
import logging
import math
import os
import sqlite3
import tempfile
import time
from collections import Counter, deque
from contextlib import closing, contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

from apps.traffic_generator.main import post_transaction
from fraud_detector.features.handbook import (
    FEATURE_VERSION,
    FeatureState,
    RunningStats,
    calculate_features,
)
from fraud_detector.schemas import BehavioralFeatures, Prediction, Transaction
from fraud_detector.simulation import (
    COLUMNS,
    GENERATOR_VERSION,
    SimulationConfig,
    TransactionWorld,
    parse_start,
)

logger = logging.getLogger(__name__)


def batch(config, start, days, output):
    if not 1 <= days <= 3650:
        raise ValueError("Days must be between 1 and 3650")
    output = Path(output)
    if output.suffix not in (".csv", ".parquet"):
        raise ValueError("Batch output must be .csv or .parquet")
    manifest = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() or manifest.exists():
        raise ValueError("Output already exists; choose a new dataset path")
    world = TransactionWorld(config, start)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts, daily = Counter(), []
    descriptor, name = tempfile.mkstemp(dir=output.parent, suffix=".partial")
    os.close(descriptor)
    temporary = Path(name)
    parquet_writer = None
    try:
        with temporary.open("w", newline="") as destination:
            writer = (
                csv.DictWriter(destination, fieldnames=COLUMNS) if output.suffix == ".csv" else None
            )
            if writer:
                writer.writeheader()
            else:
                try:
                    import pyarrow as pa
                    import pyarrow.parquet as pq
                except ImportError:
                    raise ValueError(
                        "Parquet output needs pyarrow; use CSV or install pyarrow"
                    ) from None
            for day in range(days):
                rows = world.day(day)
                positives = sum(row["TX_FRAUD"] for row in rows)
                daily.append({"day": day, "rows": len(rows), "fraud": positives})
                counts.update(str(row["TX_FRAUD_SCENARIO"]) for row in rows)
                if writer:
                    writer.writerows(rows)
                elif rows:
                    table = pa.Table.from_pylist(rows)
                    if parquet_writer is None:
                        parquet_writer = pq.ParquetWriter(str(temporary), table.schema)
                    parquet_writer.write_table(table)
            if parquet_writer:
                parquet_writer.close()
                parquet_writer = None
            destination.flush()
            os.fsync(destination.fileno())
        if not sum(counts.values()):
            raise ValueError("No transactions generated; increase population or duration")
        digest = hashlib.sha256()
        with temporary.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        report = {
            "synthetic": True,
            "generator_version": GENERATOR_VERSION,
            "config": config.model_dump(),
            "start": start.isoformat(),
            "days": days,
            "rows": sum(counts.values()),
            "fraud": sum(v for k, v in counts.items() if k != "0"),
            "scenario_counts": dict(counts),
            "daily": daily,
            "sha256": digest.hexdigest(),
        }
        os.link(temporary, output)  # Atomic publication, refusing concurrent overwrites.
        with manifest.open("x") as target:
            json.dump(report, target, indent=2)
            target.write("\n")
        return report
    finally:
        if parquet_writer:
            parquet_writer.close()
        temporary.unlink(missing_ok=True)


@contextmanager
def state_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another simulator owns this state file") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def encode_features(state):
    return {
        "customer_stats": {key: asdict(value) for key, value in state.customer_stats.items()},
        "terminal_stats": {key: asdict(value) for key, value in state.terminal_stats.items()},
        "customer_terminals": state.customer_terminals,
        "terminal_recent": {
            key: [stamp.isoformat() for stamp in stamps]
            for key, stamps in state.terminal_recent.items()
        },
        "latest_timestamp": state.latest_timestamp.isoformat() if state.latest_timestamp else None,
        "first": {key: value.isoformat() for key, value in state.first_seen.items()},
        "recent": {
            key: [value.isoformat() for value in values]
            for key, values in state.recent_transactions.items()
        },
    }


def decode_features(value):
    return FeatureState(
        customer_stats={
            key: RunningStats(**stats) for key, stats in value.get("customer_stats", {}).items()
        },
        terminal_stats={
            key: RunningStats(**stats) for key, stats in value.get("terminal_stats", {}).items()
        },
        customer_terminals=value.get("customer_terminals", {}),
        terminal_recent={
            key: deque(datetime.fromisoformat(stamp) for stamp in stamps)
            for key, stamps in value.get("terminal_recent", {}).items()
        },
        latest_timestamp=datetime.fromisoformat(value["latest_timestamp"])
        if value.get("latest_timestamp")
        else None,
        first_seen={key: datetime.fromisoformat(stamp) for key, stamp in value["first"].items()},
        recent_transactions={
            key: deque(datetime.fromisoformat(stamp) for stamp in stamps)
            for key, stamps in value["recent"].items()
        },
    )


class Stream:
    """Single producer with a durable pending request and an independent truth/response journal."""

    def __init__(self, path, config, target, start=None):
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events (
                transaction_id TEXT PRIMARY KEY, raw_json TEXT NOT NULL,
                request_json TEXT NOT NULL, response_json TEXT, delivered_at TEXT
            );
            CREATE INDEX IF NOT EXISTS events_pending ON events(delivered_at);
        """)
        try:
            saved = self.db.execute("SELECT value FROM state WHERE id=1").fetchone()
            if saved:
                self.state = json.loads(saved[0])
                if (
                    self.state["config"] != config.model_dump()
                    or self.state["target"] != target
                    or self.state["version"] != GENERATOR_VERSION
                    or (start and self.state["start"] != start.isoformat())
                ):
                    raise ValueError(
                        "State belongs to a different configuration/start/target; choose a new state file"
                    )
            else:
                start = start or datetime.now(timezone.utc).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                self.state = {
                    "version": GENERATOR_VERSION,
                    "config": config.model_dump(),
                    "target": target,
                    "start": start.isoformat(),
                    "run_id": uuid4().hex[:16],
                    "day": 0,
                    "offset": 0,
                    "features": {"first": {}, "recent": {}},
                }
                with self.db:
                    self._save()
            if self.state.get("feature_version", 1) > FEATURE_VERSION:
                raise ValueError("State uses a newer feature version")
            if self.state.get("feature_version", 1) < FEATURE_VERSION:
                # Replay local unlabeled history once; don't invent missing v2 aggregates.
                restored = FeatureState()
                for (raw_json,) in self.db.execute("SELECT raw_json FROM events ORDER BY rowid"):
                    raw = json.loads(raw_json)
                    calculate_features(
                        customer_id=str(raw["CUSTOMER_ID"]),
                        terminal_id=str(raw["TERMINAL_ID"]),
                        amount=raw["TX_AMOUNT"],
                        timestamp=datetime.fromisoformat(raw["TX_DATETIME"].replace("Z", "+00:00")),
                        state=restored,
                    )
                self.state["features"] = encode_features(restored)
                self.state["feature_version"] = FEATURE_VERSION
                with self.db:
                    self._save()
            self.features = decode_features(self.state["features"])
            self.world = TransactionWorld(config, parse_start(self.state["start"]))
            self.cached_day, self.rows = None, []
        except Exception:
            self.db.close()
            raise

    def _save(self):
        self.db.execute("INSERT OR REPLACE INTO state VALUES (1,?)", (json.dumps(self.state),))

    def pending(self):
        existing = self.db.execute(
            "SELECT request_json FROM events WHERE delivered_at IS NULL LIMIT 1"
        ).fetchone()
        if existing:
            return json.loads(existing[0])
        while True:
            day = self.state["day"]
            if day != self.cached_day:
                self.rows, self.cached_day = self.world.day(day), day
            if self.state["offset"] < len(self.rows):
                break
            self.state["day"] += 1
            self.state["offset"] = 0
        raw = self.rows[self.state["offset"]]
        stamp = datetime.fromisoformat(raw["TX_DATETIME"].replace("Z", "+00:00"))
        features = calculate_features(
            customer_id=str(raw["CUSTOMER_ID"]),
            terminal_id=str(raw["TERMINAL_ID"]),
            amount=raw["TX_AMOUNT"],
            timestamp=stamp,
            state=self.features,
        )
        request = Transaction(
            transaction_id=f"{self.state['run_id']}:{raw['TRANSACTION_ID']:014d}",
            source_transaction_id=str(raw["TRANSACTION_ID"]),
            customer_id=str(raw["CUSTOMER_ID"]),
            terminal_id=str(raw["TERMINAL_ID"]),
            amount=raw["TX_AMOUNT"],
            timestamp=stamp,
            transactions_last_hour=int(features["transactions_last_hour"]),
            customer_history_days=features["customer_history_days"],
            hour_of_day=int(features["hour_of_day"]),
            behavioral_features=BehavioralFeatures.from_features(features),
        ).model_dump(mode="json", exclude_none=True)
        self.state["offset"] += 1
        self.state["features"] = encode_features(self.features)
        with self.db:
            self.db.execute(
                "INSERT INTO events(transaction_id,raw_json,request_json) VALUES (?,?,?)",
                (request["transaction_id"], json.dumps(raw), json.dumps(request)),
            )
            self._save()
        return request

    def acknowledge(self, request, response):
        response = Prediction.model_validate(response).model_dump(mode="json")
        if response.get("transaction_id") != request["transaction_id"]:
            raise ValueError("API response transaction ID does not match the pending request")
        with self.db:
            self.db.execute(
                "UPDATE events SET response_json=?, delivered_at=? WHERE transaction_id=?",
                (
                    json.dumps(response),
                    datetime.now(timezone.utc).isoformat(),
                    request["transaction_id"],
                ),
            )

    def close(self):
        self.db.close()


def stream(config, path, target, client, start=None, count=0, rate=4.0, sleep=time.sleep):
    if not math.isfinite(rate) or not 0 < rate <= 10:
        raise ValueError("Rate must be greater than zero and at most 10 transactions/second")
    if count < 0:
        raise ValueError("Count must be nonnegative (zero means run until stopped)")
    path = Path(path)
    with state_lock(path):
        producer = Stream(path, config, target, start)
        try:
            delivered = 0
            while count == 0 or delivered < count:
                request = producer.pending()
                response = post_transaction(client, request)
                producer.acknowledge(request, response)
                delivered += 1
                if delivered % 100 == 0:
                    logger.info(
                        "simulation_progress delivered=%d simulated_day=%d",
                        delivered,
                        producer.state["day"],
                    )
                if count == 0 or delivered < count:
                    sleep(1 / rate)
            return delivered
        finally:
            producer.close()


def export_labels(state, output):
    """Export only acknowledged events; IDs directly match the API ledger's transaction IDs."""
    with (
        closing(
            sqlite3.connect(Path(state).resolve().as_uri() + "?mode=ro", uri=True)
        ) as connection,
        Path(output).open("x", newline="") as target,
    ):
        writer = csv.writer(target)
        writer.writerow(
            [
                "transaction_id",
                "source_transaction_id",
                "is_fraud",
                "fraud_scenario",
                "predicted_fraud",
                "fraud_probability",
                "model_version",
            ]
        )
        for transaction_id, raw_json, response_json in connection.execute(
            "SELECT transaction_id,raw_json,response_json FROM events WHERE delivered_at IS NOT NULL ORDER BY transaction_id"
        ):
            raw, response = json.loads(raw_json), json.loads(response_json)
            writer.writerow(
                [
                    transaction_id,
                    raw["TRANSACTION_ID"],
                    raw["TX_FRAUD"],
                    raw["TX_FRAUD_SCENARIO"],
                    response["is_fraud"],
                    response["fraud_probability"],
                    response["model_version"],
                ]
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("batch", "stream"):
        sub = commands.add_parser(name)
        sub.add_argument(
            "--config",
            type=Path,
            help="JSON SimulationConfig; unspecified fields use demo defaults",
        )
        sub.add_argument("--seed", type=int, help="Override the config seed")
        sub.add_argument(
            "--start", help="First simulated date YYYY-MM-DD (UTC); persisted on stream resume"
        )
        if name == "batch":
            sub.add_argument("--days", type=int, default=90)
            sub.add_argument("--output", type=Path, required=True)
        else:
            sub.add_argument("--state", type=Path, default=Path("data/simulator/live.db"))
            sub.add_argument("--url", default="http://localhost:8000")
            sub.add_argument(
                "--api-key-file",
                type=Path,
                default=Path(os.getenv("API_KEY_FILE", "secrets/api_key")),
            )
            sub.add_argument(
                "--rate",
                type=float,
                default=4.0,
                help="Maximum delivery rate; event time advances independently",
            )
            sub.add_argument(
                "--count",
                type=int,
                default=0,
                help="Events this invocation; zero runs until Ctrl+C",
            )
    export = commands.add_parser("export-labels")
    export.add_argument("--state", type=Path, default=Path("data/simulator/live.db"))
    export.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    try:
        if args.command == "export-labels":
            export_labels(args.state, args.output)
            print(f"Exported acknowledged truth/prediction pairs to {args.output}")
            return
        values = json.loads(args.config.read_text()) if args.config else {}
        if args.seed is not None:
            values["seed"] = args.seed
        config = SimulationConfig.model_validate(values)
        start = parse_start(args.start) if args.start else None
        if args.command == "batch":
            report = batch(config, start or parse_start("2024-01-01"), args.days, args.output)
            print(
                json.dumps(
                    {key: value for key, value in report.items() if key != "daily"}, indent=2
                )
            )
        else:
            target = args.url.rstrip("/")
            url = urlsplit(target)
            if (
                url.scheme not in ("http", "https")
                or not url.hostname
                or url.username
                or url.password
                or url.query
                or url.fragment
                or url.path
            ):
                raise ValueError(
                    "URL must be an HTTP(S) origin without credentials, path, query or fragment"
                )
            key = args.api_key_file.read_text().strip()
            if len(key) < 32:
                raise ValueError("API key must contain at least 32 characters")
            with httpx.Client(
                base_url=target,
                headers={"Authorization": f"Bearer {key}"},
                timeout=5,
                trust_env=False,
            ) as client:
                total = stream(config, args.state, target, client, start, args.count, args.rate)
            print(f"Delivered {total} transactions; resume using {args.state}")
    except KeyboardInterrupt:
        print(
            "Stopped; pending requests and history are checkpointed. Use the same command to resume."
        )
    except (ValueError, OSError, RuntimeError, sqlite3.Error) as error:
        parser.exit(1, f"Simulation stopped: {error}\n")


if __name__ == "__main__":
    main()
