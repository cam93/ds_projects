"""Private, authenticated, idempotent prediction ledger with PostgreSQL production and SQLite development backends."""

import os
import sqlite3
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request, Response

from apps.audit_store.postgres import PostgresLedger, database_url
from fraud_detector.schemas import AuditRecord
from fraud_detector.security import ResourceLimits, authorize, secret

DATABASE_PATH = os.getenv("SQLITE_PATH", "audit.db")


@contextmanager
def connect():
    connection = sqlite3.connect(DATABASE_PATH, timeout=1)
    connection.execute("PRAGMA synchronous=FULL")
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def initialize_database():
    Path(DATABASE_PATH).parent.mkdir(parents=True, exist_ok=True)
    with connect() as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("""CREATE TABLE IF NOT EXISTS predictions (
            transaction_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL,
            fraud_probability REAL NOT NULL, is_fraud INTEGER NOT NULL,
            model_version TEXT NOT NULL, source_transaction_id TEXT,
            request_hash TEXT, record_json TEXT, created_at TEXT)""")
        columns = {row[1] for row in connection.execute("PRAGMA table_info(predictions)")}
        for column in ("source_transaction_id", "request_hash", "record_json", "created_at"):
            if column not in columns:
                connection.execute(f"ALTER TABLE predictions ADD COLUMN {column} TEXT")
        connection.execute(
            "CREATE INDEX IF NOT EXISTS predictions_created ON predictions(created_at)"
        )


@asynccontextmanager
async def lifespan(app):
    app.state.secrets = {name: secret(name) for name in ("AUDIT_WRITE_KEY", "AUDIT_READ_KEY")}
    if len(set(app.state.secrets.values())) != 2:
        raise RuntimeError("Use distinct audit read and write credentials")
    app.state.postgres = None
    url = database_url()
    if url:
        app.state.postgres = PostgresLedger(url)
    elif os.getenv("APP_ENV", "production") == "production":
        raise RuntimeError("Production requires a shared HA PostgreSQL ledger")
    else:
        initialize_database()
    try:
        yield
    finally:
        if app.state.postgres:
            app.state.postgres.close()
        app.state.postgres = None


app = FastAPI(
    title="Private Prediction Ledger",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
app.add_middleware(ResourceLimits)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if getattr(app.state, "postgres", None):
        app.state.postgres.probe()
        return {"status": "ready"}
    try:
        with connect() as connection:
            connection.execute("SELECT transaction_id FROM predictions LIMIT 1")
    except sqlite3.Error:
        raise HTTPException(503, "Ledger unavailable")
    return {"status": "ready"}


@app.get("/predictions/{transaction_id}")
def get_prediction(transaction_id: str, request: Request):
    authorize(request, "AUDIT_WRITE_KEY")
    if request.app.state.postgres:
        return request.app.state.postgres.get(transaction_id)
    with connect() as connection:
        row = connection.execute(
            "SELECT record_json FROM predictions WHERE transaction_id=?", (transaction_id,)
        ).fetchone()
    if row is None:
        raise HTTPException(404, "Not found")
    if row[0] is None:
        raise HTTPException(409, "Legacy record cannot be safely replayed; use a new run ID")
    return AuditRecord.model_validate_json(row[0])


@app.post("/predictions", response_model=AuditRecord, status_code=201)
def record_prediction(record: AuditRecord, request: Request, response: Response):
    authorize(request, "AUDIT_WRITE_KEY")
    if request.app.state.postgres:
        saved, inserted = request.app.state.postgres.put(record)
        response.status_code = 201 if inserted else 200
        return saved
    try:
        with connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT request_hash, record_json FROM predictions WHERE transaction_id=?",
                (record.transaction_id,),
            ).fetchone()
            if existing:
                if existing[0] != record.request_hash or existing[1] is None:
                    raise HTTPException(409, "Transaction ID already used for different input")
                response.status_code = 200
                return AuditRecord.model_validate_json(existing[1])
            connection.execute(
                """INSERT INTO predictions
                (transaction_id,timestamp,fraud_probability,is_fraud,model_version,source_transaction_id,request_hash,record_json,created_at)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    record.transaction_id,
                    record.timestamp.isoformat(),
                    record.fraud_probability,
                    int(record.is_fraud),
                    record.model_version,
                    record.source_transaction_id,
                    record.request_hash,
                    record.model_dump_json(),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
    except sqlite3.OperationalError:
        raise HTTPException(503, "Ledger temporarily unavailable")
    return record


@app.get("/predictions")
def list_predictions(request: Request, limit: int = Query(100, ge=1, le=1000), after: str = ""):
    authorize(request, "AUDIT_READ_KEY")
    if request.app.state.postgres:
        return request.app.state.postgres.list(after, limit)
    with connect() as connection:
        rows = connection.execute(
            "SELECT record_json FROM predictions WHERE transaction_id>? AND record_json IS NOT NULL ORDER BY transaction_id LIMIT ?",
            (after, limit),
        ).fetchall()
    return [AuditRecord.model_validate_json(row[0]) for row in rows]
