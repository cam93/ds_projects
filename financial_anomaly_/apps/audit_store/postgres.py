"""Shared ledger backend: PostgreSQL uniqueness is the cross-replica arbiter."""

import os
from contextlib import contextmanager
from pathlib import Path

import psycopg
from fastapi import HTTPException
from psycopg.conninfo import conninfo_to_dict
from psycopg_pool import ConnectionPool, PoolTimeout, TooManyRequests

from fraud_detector.schemas import AuditRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    transaction_id VARCHAR(128) PRIMARY KEY,
    source_transaction_id VARCHAR(128),
    timestamp TIMESTAMPTZ NOT NULL,
    fraud_probability DOUBLE PRECISION NOT NULL CHECK (fraud_probability BETWEEN 0 AND 1),
    is_fraud BOOLEAN NOT NULL,
    model_version VARCHAR(128) NOT NULL,
    request_hash CHAR(64) NOT NULL,
    record_json TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS predictions_created ON predictions(created_at);
"""


def database_url():
    filename = os.getenv("DATABASE_URL_FILE")
    return Path(filename).read_text().strip() if filename else os.getenv("DATABASE_URL", "")


def validate_connection_url(url):
    settings = conninfo_to_dict(url)
    if (
        os.getenv("APP_ENV", "production") == "production"
        and settings.get("sslmode") != "verify-full"
    ):
        raise ValueError("Production PostgreSQL requires sslmode=verify-full and a trusted CA")
    return url


class PostgresLedger:
    def __init__(self, url):
        validate_connection_url(url)
        self.pool = ConnectionPool(
            url,
            min_size=1,
            max_size=16,
            timeout=2,
            max_waiting=32,
            max_lifetime=300,
            check=ConnectionPool.check_connection,
            open=False,
            kwargs={
                "connect_timeout": 3,
                "options": "-c statement_timeout=2000 -c lock_timeout=1000 -c synchronous_commit=on",
            },
        )
        self.pool.open(wait=True, timeout=10)
        try:
            self.probe()
        except Exception:
            self.pool.close()
            raise

    @contextmanager
    def connection(self):
        try:
            with self.pool.connection() as connection:
                yield connection
        except (psycopg.Error, PoolTimeout, TooManyRequests):
            raise HTTPException(503, "Ledger temporarily unavailable") from None

    def probe(self):
        with self.connection() as connection:
            connection.execute("SELECT transaction_id FROM predictions LIMIT 1")
            if connection.execute("SELECT pg_is_in_recovery()").fetchone()[0]:
                raise HTTPException(503, "Ledger primary unavailable")

    def get(self, transaction_id):
        with self.connection() as connection:
            row = connection.execute(
                "SELECT record_json FROM predictions WHERE transaction_id=%s", (transaction_id,)
            ).fetchone()
        if row is None:
            raise HTTPException(404, "Not found")
        return AuditRecord.model_validate_json(row[0])

    def put(self, record):
        with self.connection() as connection:
            inserted = connection.execute(
                """INSERT INTO predictions
                (transaction_id,source_transaction_id,timestamp,fraud_probability,is_fraud,model_version,request_hash,record_json)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (transaction_id) DO NOTHING
                RETURNING transaction_id""",
                (
                    record.transaction_id,
                    record.source_transaction_id,
                    record.timestamp,
                    record.fraud_probability,
                    record.is_fraud,
                    record.model_version,
                    record.request_hash,
                    record.model_dump_json(),
                ),
            ).fetchone()
            # READ COMMITTED: this statement sees a concurrent insert after its commit.
            row = connection.execute(
                "SELECT request_hash,record_json FROM predictions WHERE transaction_id=%s",
                (record.transaction_id,),
            ).fetchone()
            if row[0] != record.request_hash:
                raise HTTPException(409, "Transaction ID already used for different input")
            result = AuditRecord.model_validate_json(row[1])
        # Acknowledgement occurs only after transaction commit (and configured synchronous replication).
        return result, bool(inserted)

    def list(self, after, limit):
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT record_json FROM predictions WHERE transaction_id>%s ORDER BY transaction_id LIMIT %s",
                (after, limit),
            ).fetchall()
        return [AuditRecord.model_validate_json(row[0]) for row in rows]

    def close(self):
        self.pool.close()
