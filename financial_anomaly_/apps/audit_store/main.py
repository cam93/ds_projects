import os
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

DATABASE_PATH = os.getenv("SQLITE_PATH", "audit.db")


class AuditRecord(BaseModel):
    transaction_id: str
    source_transaction_id: str | None = None
    timestamp: datetime
    fraud_probability: float = Field(ge=0, le=1)
    is_fraud: bool
    model_version: str


def initialize_database() -> None:
    database_directory = os.path.dirname(DATABASE_PATH)
    if database_directory:
        os.makedirs(database_directory, exist_ok=True)
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS predictions (
                transaction_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                fraud_probability REAL NOT NULL,
                is_fraud INTEGER NOT NULL,
                model_version TEXT NOT NULL
                ,source_transaction_id TEXT
            )"""
        )
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(predictions)")
        }
        if "source_transaction_id" not in columns:
            connection.execute("ALTER TABLE predictions ADD COLUMN source_transaction_id TEXT")


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_database()
    yield


app = FastAPI(title="Fraud Prediction Audit Store", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predictions", status_code=201)
def record_prediction(record: AuditRecord) -> AuditRecord:
    try:
        with sqlite3.connect(DATABASE_PATH) as connection:
            connection.execute(
                "INSERT INTO predictions "
                "(transaction_id, timestamp, fraud_probability, is_fraud, model_version, source_transaction_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    record.transaction_id,
                    record.timestamp.isoformat(),
                    record.fraud_probability,
                    int(record.is_fraud),
                    record.model_version,
                    record.source_transaction_id,
                ),
            )
    except sqlite3.IntegrityError as error:
        raise HTTPException(status_code=409, detail="transaction already audited") from error
    return record


@app.get("/predictions")
def list_predictions(limit: Annotated[int, Field(ge=1, le=100)] = 20) -> list[AuditRecord]:
    with sqlite3.connect(DATABASE_PATH) as connection:
        rows = connection.execute(
            "SELECT transaction_id, source_transaction_id, timestamp, fraud_probability, is_fraud, model_version "
            "FROM predictions ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        AuditRecord(
            transaction_id=row[0],
            source_transaction_id=row[1],
            timestamp=datetime.fromisoformat(row[2]),
            fraud_probability=row[3],
            is_fraud=bool(row[4]),
            model_version=row[5],
        )
        for row in rows
    ]
