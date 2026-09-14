from pathlib import Path

import pandas as pd
import pytest

from scripts.prepare_dataset import prepare_dataset, validate_and_sort


def test_validation_sorts_by_timestamp_then_transaction_id() -> None:
    records = pd.DataFrame(
        [
            {"TRANSACTION_ID": 2, "TX_DATETIME": "2020-01-01 00:00:01", "TX_AMOUNT": 2},
            {"TRANSACTION_ID": 1, "TX_DATETIME": "2020-01-01 00:00:01", "TX_AMOUNT": 1},
        ]
    )
    for column, value in {
        "CUSTOMER_ID": 1,
        "TERMINAL_ID": 1,
        "TX_FRAUD": 0,
        "TX_FRAUD_SCENARIO": 0,
    }.items():
        records[column] = value

    prepared = validate_and_sort(records)

    assert prepared["TRANSACTION_ID"].tolist() == ["1", "2"]


def test_validation_rejects_duplicate_ids() -> None:
    records = pd.DataFrame(
        {
            "TRANSACTION_ID": [1, 1],
            "TX_DATETIME": ["2020-01-01", "2020-01-02"],
            "TX_AMOUNT": [1, 2],
            "CUSTOMER_ID": [1, 2],
            "TERMINAL_ID": [1, 2],
            "TX_FRAUD": [0, 0],
            "TX_FRAUD_SCENARIO": [0, 0],
        }
    )

    with pytest.raises(ValueError, match="unique"):
        validate_and_sort(records)


def test_prepare_dataset_writes_three_outputs(tmp_path: Path) -> None:
    source = tmp_path / "source.pkl"
    pd.DataFrame(
        {
            "TRANSACTION_ID": [1],
            "TX_DATETIME": ["2020-01-01"],
            "CUSTOMER_ID": [1],
            "TERMINAL_ID": [1],
            "TX_AMOUNT": [10.0],
            "TX_FRAUD": [0],
            "TX_FRAUD_SCENARIO": [0],
        }
    ).to_pickle(source)

    output_dir = tmp_path / "processed"
    prepare_dataset(source, output_dir)

    assert (output_dir / "records.csv").exists()
    assert (output_dir / "training_features.csv").exists()
    assert (output_dir / "replay_events.jsonl").exists()
