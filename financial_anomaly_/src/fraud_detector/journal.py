"""Durable simulator journal, feature-state migration and acknowledgement handling."""

import fcntl
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from uuid import uuid4

from fraud_detector.features.adaptive import AdaptiveState
from fraud_detector.features.checkpoint import decode_features, encode_features
from fraud_detector.features.handbook import FEATURE_VERSION, FeatureState, calculate_features
from fraud_detector.features.pipeline import enrich_transaction, prepared_transaction
from fraud_detector.quality import add_quality, initialize_quality
from fraud_detector.schemas import Prediction
from fraud_detector.simulation import GENERATOR_VERSION, TransactionWorld, parse_start


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


class Stream:
    """Single producer with a durable pending request and an independent truth/response journal."""

    def __init__(self, path, config, target, start=None, feedback_delay_days=7):
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
            initialize_quality(self.db)
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
            if "adaptive" not in self.state:
                adaptive = AdaptiveState(feedback_delay_days)
                for (raw_json,) in self.db.execute("SELECT raw_json FROM events ORDER BY rowid"):
                    adaptive.observe_raw(json.loads(raw_json))
                self.state["adaptive"] = adaptive.snapshot()
                with self.db:
                    self._save()
            if self.state["adaptive"]["delay_days"] != feedback_delay_days:
                raise ValueError("Feedback delay differs from saved state; use a new state file")
            self.adaptive = AdaptiveState(feedback_delay_days, self.state["adaptive"])
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
        features = enrich_transaction(raw, self.features, self.adaptive)
        request = prepared_transaction(
            transaction_id=f"{self.state['run_id']}:{raw['TRANSACTION_ID']:014d}",
            source_transaction_id=str(raw["TRANSACTION_ID"]),
            customer_id=raw["CUSTOMER_ID"],
            terminal_id=raw["TERMINAL_ID"],
            timestamp=stamp,
            features=features,
            feedback_delay_days=self.adaptive.delay_days,
        ).model_dump(mode="json", exclude_none=True)
        self.state["adaptive"] = self.adaptive.snapshot()
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
            event = self.db.execute(
                "SELECT raw_json FROM events WHERE transaction_id=? AND delivered_at IS NULL",
                (request["transaction_id"],),
            ).fetchone()
            if event is None:
                return  # A duplicate acknowledgement must not count twice.
            add_quality(self.db, json.loads(event[0]), response)
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
