"""Authenticated aggregate demo-oracle metrics; no transaction or customer identifiers."""

import hmac
import json
import sqlite3
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from prometheus_client import CollectorRegistry, generate_latest
from prometheus_client.core import GaugeMetricFamily


def add_quality(db, raw, response):
    actual, predicted = bool(raw["TX_FRAUD"]), bool(response["is_fraud"])
    outcome = ("tp" if predicted else "fn") if actual else ("fp" if predicted else "tn")
    db.execute(
        "INSERT INTO quality VALUES (?,?,?,1) ON CONFLICT(model_version,scenario,outcome) DO UPDATE SET count=count+1",
        (response["model_version"], str(raw["TX_FRAUD_SCENARIO"]), outcome),
    )


def initialize_quality(db):
    exists = db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='quality'"
    ).fetchone()
    if exists:
        return
    db.execute("BEGIN IMMEDIATE")
    with db:
        db.execute(
            "CREATE TABLE quality(model_version TEXT, scenario TEXT, outcome TEXT, count INTEGER NOT NULL, PRIMARY KEY(model_version,scenario,outcome))"
        )
        for raw, response in db.execute(
            "SELECT raw_json,response_json FROM events WHERE delivered_at IS NOT NULL"
        ):
            add_quality(db, json.loads(raw), json.loads(response))


class QualityCollector:
    def __init__(self, path):
        self.path = Path(path).resolve()

    def collect(self):
        values = GaugeMetricFamily(
            "fraud_demo_quality",
            "Cumulative acknowledged synthetic oracle outcomes, not delayed real investigations",
            labels=["model_version", "scenario", "outcome"],
        )
        with closing(sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=2)) as db:
            rows = db.execute("SELECT model_version,scenario,outcome,count FROM quality").fetchall()
        groups = {(version, scenario) for version, scenario, _, _ in rows}
        counts = {(version, scenario, outcome): count for version, scenario, outcome, count in rows}
        for version, scenario in sorted(groups):
            for outcome in ("tp", "fp", "tn", "fn"):
                values.add_metric(
                    [version, scenario, outcome], counts.get((version, scenario, outcome), 0)
                )
        yield values


def start_metrics(path, port, key, host="0.0.0.0"):
    if len(key) < 32:
        raise ValueError("Metrics key must contain at least 32 characters")
    registry = CollectorRegistry()
    registry.register(QualityCollector(path))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/metrics":
                self.send_error(404)
                return
            if not hmac.compare_digest(
                self.headers.get("Authorization", "").encode(), ("Bearer " + key).encode()
            ):
                self.send_error(401)
                return
            try:
                contents = generate_latest(registry)
            except sqlite3.Error:
                self.send_error(503, "Quality journal unavailable")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(contents)))
            self.end_headers()
            self.wfile.write(contents)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    Thread(target=server.serve_forever, daemon=True).start()
    return server
