"""Evaluate a single replay run against held-out labels using a ledger backup."""

import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd
from train import classification_metrics


def evaluate(database, labels_path, run_id, postgres_url_file=None):
    labels = pd.read_csv(labels_path, dtype={"transaction_id": str})
    if (
        labels.empty
        or labels.transaction_id.duplicated().any()
        or not labels.is_fraud.isin([0, 1]).all()
    ):
        raise ValueError("Labels must be non-empty, binary, and unique")
    query = "SELECT transaction_id, source_transaction_id, fraud_probability, is_fraud AS predicted_fraud, model_version FROM predictions"
    if postgres_url_file:
        import psycopg
        from psycopg.rows import dict_row

        from apps.audit_store.postgres import validate_connection_url

        url = validate_connection_url(Path(postgres_url_file).read_text().strip())
        with psycopg.connect(url, row_factory=dict_row, connect_timeout=5) as connection:
            connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            connection.execute("SET LOCAL statement_timeout = '60s'")
            rows = connection.execute(
                query + " WHERE left(transaction_id,%s)=%s",
                (len(run_id) + 1, run_id + ":"),
            ).fetchall()
            predictions = pd.DataFrame(
                rows,
                columns=[
                    "transaction_id",
                    "source_transaction_id",
                    "fraud_probability",
                    "predicted_fraud",
                    "model_version",
                ],
            )
    else:
        with sqlite3.connect(f"file:{Path(database).resolve()}?mode=ro", uri=True) as connection:
            predictions = pd.read_sql_query(
                query + " WHERE substr(transaction_id,1,?)=?",
                connection,
                params=(len(run_id) + 1, run_id + ":"),
            )
    if len(predictions) != len(labels) or predictions.source_transaction_id.duplicated().any():
        raise ValueError("Replay is incomplete or has duplicate/extra source IDs")
    joined = labels.merge(
        predictions,
        left_on="transaction_id",
        right_on="source_transaction_id",
        validate="one_to_one",
        how="left",
    )
    if joined.fraud_probability.isna().any() or joined.model_version.nunique() != 1:
        raise ValueError("Missing predictions or mixed model versions")
    if not 0 < joined.is_fraud.sum() < len(joined):
        raise ValueError("Evaluation requires both classes")
    # Decisions use the actual deployed threshold; AP uses the original scores.
    metrics = classification_metrics(joined.is_fraud.tolist(), joined.predicted_fraud.tolist(), 0.5)
    from train import average_precision

    metrics["average_precision"] = average_precision(
        joined.is_fraud.tolist(), joined.fraud_probability.tolist()
    )
    return {"rows": len(joined), "model_version": joined.model_version.iloc[0], "metrics": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--postgres-url-file",
        type=Path,
        help="Read-only PostgreSQL credential file; database positional argument is ignored",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(args.database, args.labels, args.run_id, args.postgres_url_file), indent=2
        )
    )


if __name__ == "__main__":
    main()
