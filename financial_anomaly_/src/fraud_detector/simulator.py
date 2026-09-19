"""Batch generation and checkpointed live delivery of synthetic Handbook transactions."""

import argparse
import csv
import json
import logging
import math
import os
import sqlite3
import tempfile
import time
from collections import Counter
from contextlib import closing
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from fraud_detector.artifacts import digest
from fraud_detector.delivery import post_transaction
from fraud_detector.features.checkpoint import decode_features, encode_features
from fraud_detector.journal import Stream, state_lock
from fraud_detector.quality import start_metrics
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
            "sha256": digest(temporary),
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


def stream(
    config,
    path,
    target,
    client,
    start=None,
    count=0,
    rate=4.0,
    sleep=time.sleep,
    feedback_delay_days=7,
):
    if not math.isfinite(rate) or not 0 < rate <= 10:
        raise ValueError("Rate must be greater than zero and at most 10 transactions/second")
    if count < 0:
        raise ValueError("Count must be nonnegative (zero means run until stopped)")
    path = Path(path)
    with state_lock(path):
        producer = Stream(path, config, target, start, feedback_delay_days)
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
            sub.add_argument("--feedback-delay-days", type=float, default=7)
            sub.add_argument(
                "--metrics-port",
                type=int,
                default=0,
                help="Optional private metrics listener; 0 disables",
            )
            sub.add_argument(
                "--metrics-key-file", type=Path, default=Path("/run/secrets/metrics_key")
            )
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
                server = (
                    start_metrics(
                        args.state, args.metrics_port, args.metrics_key_file.read_text().strip()
                    )
                    if args.metrics_port
                    else None
                )
                try:
                    total = stream(
                        config,
                        args.state,
                        target,
                        client,
                        start,
                        args.count,
                        args.rate,
                        feedback_delay_days=args.feedback_delay_days,
                    )
                finally:
                    if server:
                        server.shutdown()
                        server.server_close()
            print(f"Delivered {total} transactions; resume using {args.state}")
    except KeyboardInterrupt:
        print(
            "Stopped; pending requests and history are checkpointed. Use the same command to resume."
        )
    except (ValueError, OSError, RuntimeError, sqlite3.Error) as error:
        parser.exit(1, f"Simulation stopped: {error}\n")


__all__ = [
    "Stream",
    "batch",
    "decode_features",
    "encode_features",
    "export_labels",
    "main",
    "state_lock",
    "stream",
]


if __name__ == "__main__":
    main()
