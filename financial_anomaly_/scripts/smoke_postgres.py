"""Verify cross-process ledger semantics against an isolated real PostgreSQL server."""

import os
import secrets
import socket
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import psycopg
from fastapi import HTTPException
from psycopg import sql

from apps.audit_store.postgres import SCHEMA, PostgresLedger
from fraud_detector.schemas import AuditRecord


def docker(*args):
    return subprocess.run(
        ["docker", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def main():
    name = "fraud-postgres-test-" + uuid4().hex[:10]
    previous = os.environ.get("APP_ENV")
    os.environ["APP_ENV"] = "development"
    pools = []
    with tempfile.TemporaryDirectory(prefix="fraud-pg-test-") as directory:
        folder = Path(directory)
        folder.chmod(0o755)
        password = secrets.token_hex(32)
        (folder / "password").write_text(password)
        (folder / "password").chmod(0o444)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        try:
            docker(
                "run",
                "-d",
                "--name",
                name,
                "-p",
                f"127.0.0.1:{port}:5432",
                "-v",
                f"{folder}/password:/run/password:ro",
                "-e",
                "POSTGRES_PASSWORD_FILE=/run/password",
                "-e",
                "POSTGRES_DB=fraud",
                "postgres:17.11-alpine@sha256:18cfe3ef5e6815560c98237d6216d1e5119702fb0f3894c8785dd58b8bbe5d73",
            )
            admin = f"postgresql://postgres:{password}@127.0.0.1:{port}/fraud"
            for _ in range(30):
                try:
                    with psycopg.connect(admin, connect_timeout=1) as connection:
                        connection.execute(SCHEMA)
                        connection.execute(
                            sql.SQL("CREATE USER fraud_app PASSWORD {}").format(
                                sql.Literal(password)
                            )
                        )
                        connection.execute("GRANT USAGE ON SCHEMA public TO fraud_app")
                        connection.execute("GRANT SELECT, INSERT ON predictions TO fraud_app")
                    break
                except psycopg.OperationalError:
                    time.sleep(1)
            else:
                raise RuntimeError("PostgreSQL did not start")
            url = f"postgresql://fraud_app:{password}@127.0.0.1:{port}/fraud"
            pools = [PostgresLedger(url), PostgresLedger(url)]
            record = AuditRecord(
                transaction_id="smoke:1",
                source_transaction_id="1",
                timestamp=datetime.now(timezone.utc),
                fraud_probability=0.25,
                is_fraud=False,
                model_version="smoke",
                request_hash="a" * 64,
            )
            with ThreadPoolExecutor(max_workers=12) as executor:
                results = list(executor.map(lambda i: pools[i % 2].put(record), range(24)))
            assert sum(inserted for _, inserted in results) == 1
            assert len(pools[0].list("", 100)) == 1
            try:
                pools[1].put(record.model_copy(update={"request_hash": "b" * 64}))
            except HTTPException as error:
                assert error.status_code == 409
            else:
                raise AssertionError("Conflicting transaction accepted")
            with psycopg.connect(url) as connection:
                try:
                    connection.execute("DELETE FROM predictions")
                except psycopg.errors.InsufficientPrivilege:
                    connection.rollback()
                else:
                    raise AssertionError("Runtime role could delete audit history")
            from evaluate_replay import evaluate

            pools[1].put(
                record.model_copy(
                    update={
                        "transaction_id": "smoke:2",
                        "source_transaction_id": "2",
                        "is_fraud": True,
                        "fraud_probability": 0.9,
                        "request_hash": "c" * 64,
                    }
                )
            )
            labels = folder / "labels.csv"
            labels.write_text("transaction_id,is_fraud\n1,0\n2,1\n")
            url_file = folder / "evaluation-url"
            url_file.write_text(url)
            url_file.chmod(0o600)
            evaluation = evaluate(None, labels, "smoke", url_file)
            assert evaluation["rows"] == 2
            assert evaluation["metrics"]["average_precision"] == 1.0
            docker("stop", "--time", "2", name)
            try:
                pools[0].probe()
            except HTTPException as error:
                assert error.status_code == 503
            else:
                raise AssertionError("Database outage not detected")
            docker("start", name)
            for _ in range(20):
                try:
                    assert pools[1].get("smoke:1") == record
                    break
                except HTTPException:
                    time.sleep(1)
            else:
                raise RuntimeError("Pool did not recover after database restart")
            print(
                "PostgreSQL concurrent idempotency, role restrictions, outage detection and durable restart recovery passed"
            )
        finally:
            for pool in pools:
                pool.close()
            subprocess.run(["docker", "rm", "-f", "-v", name], capture_output=True, check=False)
            if previous is None:
                os.environ.pop("APP_ENV", None)
            else:
                os.environ["APP_ENV"] = previous


if __name__ == "__main__":
    main()
