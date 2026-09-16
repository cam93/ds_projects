"""Validate and materialize the fraud-simulation dataset."""

import argparse
import hashlib
import io
import json
import math
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from train import split_by_time

from fraud_detector.features.handbook import V2_FEATURE_NAMES, FeatureState, calculate_features
from fraud_detector.schemas import BehavioralFeatures, Transaction

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

    if records.empty:
        raise ValueError("Dataset is empty")
    prepared = records.copy()
    for name in ("TRANSACTION_ID", "CUSTOMER_ID", "TERMINAL_ID"):
        prepared[name] = prepared[name].astype("string")
        if (
            prepared[name].isna().any()
            or not prepared[name].str.fullmatch(r"[A-Za-z0-9_.:-]{1,96}").all()
        ):
            raise ValueError(f"Invalid {name}")
    for name, allowed in (("TX_FRAUD", [0, 1]), ("TX_FRAUD_SCENARIO", [0, 1, 2, 3])):
        prepared[name] = pd.to_numeric(prepared[name], errors="raise")
        if not prepared[name].isin(allowed).all():
            raise ValueError(f"Invalid {name}")
    if prepared["TX_AMOUNT"].apply(lambda v: isinstance(v, bool)).any():
        raise ValueError("Amounts cannot be booleans")
    prepared["TRANSACTION_ID"] = prepared["TRANSACTION_ID"].astype("string")
    prepared["TX_DATETIME"] = pd.to_datetime(prepared["TX_DATETIME"], errors="coerce", utc=True)
    prepared["TX_AMOUNT"] = pd.to_numeric(prepared["TX_AMOUNT"], errors="coerce")

    errors: list[str] = []
    if prepared["TRANSACTION_ID"].isna().any() or (prepared["TRANSACTION_ID"].str.len() == 0).any():
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
    if prepared["TX_AMOUNT"].gt(1_000_000).any():
        errors.append("amount exceeds API limit")
    if errors:
        raise ValueError("Dataset validation failed: " + "; ".join(errors))

    return prepared.sort_values(
        by=["TX_DATETIME", "TRANSACTION_ID"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_training_features(records: pd.DataFrame) -> pd.DataFrame:
    """Create model-ready numeric features while retaining the fraud label."""
    online_features = build_online_features(records)
    return pd.DataFrame(
        {
            "transaction_id": records["TRANSACTION_ID"],
            "timestamp": records["TX_DATETIME"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "customer_id": records["CUSTOMER_ID"].astype("string"),
            "terminal_id": records["TERMINAL_ID"].astype("string"),
            **online_features,
            "is_fraud": pd.to_numeric(records["TX_FRAUD"], errors="raise").astype("int64"),
            "fraud_scenario": pd.to_numeric(records["TX_FRAUD_SCENARIO"], errors="raise").astype(
                "int64"
            ),
        }
    )


def build_online_features(records: pd.DataFrame) -> pd.DataFrame:
    """Calculate handbook features in chronological order."""
    state = FeatureState()
    calculated: list[dict[str, float]] = []
    for index, row in records.iterrows():
        calculated.append(
            calculate_features(
                customer_id=str(row["CUSTOMER_ID"]),
                terminal_id=str(row["TERMINAL_ID"]),
                amount=float(row["TX_AMOUNT"]),
                timestamp=row["TX_DATETIME"].to_pydatetime(),
                state=state,
            )
        )
    return pd.DataFrame(calculated, index=records.index)[V2_FEATURE_NAMES]


def build_replay_events(records: pd.DataFrame) -> list[dict[str, object]]:
    """Create events accepted by the realtime API in deterministic order."""
    online_features = build_online_features(records)
    events: list[dict[str, object]] = []
    for position, record in enumerate(records.itertuples(index=False)):
        timestamp = pd.Timestamp(record.TX_DATETIME).isoformat().replace("+00:00", "Z")
        events.append(
            {
                "transaction_id": str(record.TRANSACTION_ID),
                "customer_id": str(record.CUSTOMER_ID),
                "terminal_id": str(record.TERMINAL_ID),
                "amount": float(record.TX_AMOUNT),
                "transactions_last_hour": int(
                    online_features.iloc[position]["transactions_last_hour"]
                ),
                "customer_history_days": float(
                    online_features.iloc[position]["customer_history_days"]
                ),
                "hour_of_day": int(online_features.iloc[position]["hour_of_day"]),
                "timestamp": timestamp,
                "behavioral_features": BehavioralFeatures.from_features(
                    online_features.iloc[position]
                ).model_dump(),
            }
        )
    return events


def write_outputs(records: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    features = build_training_features(records)
    records.to_csv(output_dir / "records.csv", index=False)
    features.to_csv(output_dir / "training_features.csv", index=False)
    # Preserve leading-zero IDs and calculate history once across all partitions.
    if len(features.timestamp.unique()) >= 7:
        _, _, test = split_by_time(features)
    else:
        test = features.iloc[0:0]
    test_ids = set(test.transaction_id)
    handbook = output_dir / "handbook"
    handbook.mkdir(exist_ok=True)
    test[["transaction_id", "is_fraud"]].to_csv(handbook / "test_labels.csv", index=False)
    with (
        (output_dir / "replay_events.jsonl").open("w") as all_file,
        (handbook / "replay.jsonl").open("w") as test_file,
    ):
        for row in features.itertuples(index=False):
            event = Transaction(
                transaction_id=str(row.transaction_id),
                customer_id=str(row.customer_id),
                terminal_id=str(row.terminal_id),
                amount=float(row.amount),
                transactions_last_hour=int(row.transactions_last_hour),
                customer_history_days=float(row.customer_history_days),
                hour_of_day=int(row.hour_of_day),
                timestamp=row.timestamp,
                behavioral_features=BehavioralFeatures.from_features(row._asdict()),
            )
            line = event.model_dump_json(exclude_none=True) + "\n"
            all_file.write(line)
            if row.transaction_id in test_ids:
                test_file.write(line)


def read_source(path: Path, trusted_manifest: dict | None = None) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(
            path, dtype={name: str for name in ("TRANSACTION_ID", "CUSTOMER_ID", "TERMINAL_ID")}
        )
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".pkl":
        expected = (trusted_manifest or {}).get(path.name)
        contents = path.read_bytes()
        if not expected or hashlib.sha256(contents).hexdigest() != expected:
            raise ValueError(
                "Pickle requires a verified SHA256 in --trusted-manifest before deserialization"
            )
        return pd.read_pickle(io.BytesIO(contents))
    raise ValueError("Supported inputs: CSV, Parquet, verified pickle")


def prepare_dataset(
    input_path: Path, output_dir: Path, trusted_manifest: dict | None = None
) -> pd.DataFrame:
    records = read_source(input_path, trusted_manifest)
    if not isinstance(records, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(records).__name__}")
    prepared = validate_and_sort(records)
    write_outputs(prepared, output_dir)
    return prepared


def prepare_pickles(
    input_paths: Sequence[Path], output_dir: Path, trusted_manifest: dict | None = None
) -> pd.DataFrame:
    """Read and combine one or more pickle files before global validation."""
    frames = []
    for input_path in input_paths:
        records = read_source(input_path, trusted_manifest)
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
    parser.add_argument(
        "--trusted-manifest",
        type=Path,
        help="Reviewed mapping of pickle filenames to trusted SHA256 digests",
    )
    return parser.parse_args(arguments)


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.trusted_manifest.read_text()) if args.trusted_manifest else None
    prepared = prepare_pickles(args.input_paths, args.output_dir, manifest)
    print(f"Validated and saved {len(prepared)} records to {args.output_dir}")


if __name__ == "__main__":
    main()
