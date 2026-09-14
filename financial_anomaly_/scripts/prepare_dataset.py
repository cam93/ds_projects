"""Validate and materialize the fraud-simulation dataset."""

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = (
    "TRANSACTION_ID",
    "TX_DATETIME",
    "CUSTOMER_ID",
    "TERMINAL_ID",
    "TX_AMOUNT",
    "TX_FRAUD",
    "TX_FRAUD_SCENARIO",
)


def validate_and_sort(records: pd.DataFrame) -> pd.DataFrame:
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(records.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    prepared = records.copy()
    prepared["TRANSACTION_ID"] = prepared["TRANSACTION_ID"].astype("string")
    prepared["TX_DATETIME"] = pd.to_datetime(
        prepared["TX_DATETIME"], errors="coerce", utc=True
    )
    prepared["TX_AMOUNT"] = pd.to_numeric(prepared["TX_AMOUNT"], errors="coerce")

    errors: list[str] = []
    if prepared["TRANSACTION_ID"].isna().any() or (
        prepared["TRANSACTION_ID"].str.len() == 0
    ).any():
        errors.append("transaction IDs must be non-empty")
    if prepared["TRANSACTION_ID"].duplicated().any():
        errors.append("transaction IDs must be unique")
    if prepared["TX_DATETIME"].isna().any():
        errors.append("timestamps must be valid and non-null")
    if prepared["TX_AMOUNT"].isna().any():
        errors.append("amounts must be numeric")
    if (~prepared["TX_AMOUNT"].map(lambda value: math.isfinite(float(value)))).any():
        errors.append("amounts must be finite")
    if (~prepared["TX_AMOUNT"].gt(0)).any():
        errors.append("amounts must be positive")
    if errors:
        raise ValueError("Dataset validation failed: " + "; ".join(errors))

    return prepared.sort_values(
        by=["TX_DATETIME", "TRANSACTION_ID"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_training_features(records: pd.DataFrame) -> pd.DataFrame:
    """Create model-ready numeric features while retaining the fraud label."""
    customer_history_days, transactions_last_hour = build_online_features(records)
    return pd.DataFrame(
        {
            "transaction_id": records["TRANSACTION_ID"],
            "timestamp": records["TX_DATETIME"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "customer_id": records["CUSTOMER_ID"].astype("string"),
            "terminal_id": records["TERMINAL_ID"].astype("string"),
            "amount": records["TX_AMOUNT"].astype("float64"),
            "transactions_last_hour": transactions_last_hour,
            "customer_history_days": customer_history_days,
            "is_fraud": pd.to_numeric(records["TX_FRAUD"], errors="raise").astype("int64"),
            "fraud_scenario": pd.to_numeric(
                records["TX_FRAUD_SCENARIO"], errors="raise"
            ).astype("int64"),
        }
    )


def build_online_features(records: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Calculate only features available at the time each transaction occurs."""
    first_seen = records.groupby("CUSTOMER_ID")["TX_DATETIME"].transform("min")
    history_days = (records["TX_DATETIME"] - first_seen).dt.total_seconds().div(86400)
    prior_transactions = pd.Series(0, index=records.index, dtype="int64")
    customer_windows: dict[str, list[pd.Timestamp]] = {}
    for index, row in records.iterrows():
        customer_id = str(row["CUSTOMER_ID"])
        timestamp = row["TX_DATETIME"]
        window = customer_windows.setdefault(customer_id, [])
        window[:] = [seen_at for seen_at in window if timestamp - seen_at < pd.Timedelta(hours=1)]
        prior_transactions.at[index] = len(window)
        window.append(timestamp)
    return history_days.round(6), prior_transactions


def build_replay_events(records: pd.DataFrame) -> list[dict[str, object]]:
    """Create events accepted by the realtime API in deterministic order."""
    customer_history_days, transactions_last_hour = build_online_features(records)
    events: list[dict[str, object]] = []
    for position, record in enumerate(records.itertuples(index=False)):
        timestamp = pd.Timestamp(record.TX_DATETIME).isoformat().replace("+00:00", "Z")
        events.append(
            {
                "transaction_id": str(record.TRANSACTION_ID),
                "customer_id": str(record.CUSTOMER_ID),
                "terminal_id": str(record.TERMINAL_ID),
                "amount": float(record.TX_AMOUNT),
                "transactions_last_hour": int(transactions_last_hour.iloc[position]),
                "customer_history_days": float(customer_history_days.iloc[position]),
                "timestamp": timestamp,
            }
        )
    return events


def write_outputs(records: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_records = records.copy()
    canonical_records["TX_DATETIME"] = canonical_records["TX_DATETIME"].dt.strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    canonical_records.to_csv(output_dir / "records.csv", index=False)
    build_training_features(records).to_csv(
        output_dir / "training_features.csv", index=False
    )
    with (output_dir / "replay_events.jsonl").open("w", encoding="utf-8") as replay_file:
        for event in build_replay_events(records):
            replay_file.write(json.dumps(event, separators=(",", ":")) + "\n")


def prepare_dataset(input_path: Path, output_dir: Path) -> pd.DataFrame:
    records = pd.read_pickle(input_path)
    if not isinstance(records, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(records).__name__}")
    prepared = validate_and_sort(records)
    write_outputs(prepared, output_dir)
    return prepared


def prepare_pickles(input_paths: Sequence[Path], output_dir: Path) -> pd.DataFrame:
    """Read and combine one or more pickle files before global validation."""
    frames = []
    for input_path in input_paths:
        records = pd.read_pickle(input_path)
        if not isinstance(records, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(records).__name__}")
        frames.append(records)
    if not frames:
        raise ValueError("At least one pickle file is required")
    prepared = validate_and_sort(pd.concat(frames, ignore_index=True))
    write_outputs(prepared, output_dir)
    return prepared


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_paths",
        type=Path,
        nargs="+",
        help="One or more pandas pickle files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for records, training features, and replay events",
    )
    return parser.parse_args(arguments)


def main() -> None:
    args = parse_args()
    prepared = prepare_pickles(args.input_paths, args.output_dir)
    print(f"Validated and saved {len(prepared)} records to {args.output_dir}")


if __name__ == "__main__":
    main()
