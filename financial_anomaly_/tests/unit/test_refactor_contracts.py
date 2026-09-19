"""Compatibility boundaries affected by moving CLI logic into reusable modules."""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

from fraud_detector.schemas import Transaction


def test_dataset_preparation_does_not_import_training_dependencies():
    code = """
import sys
from importlib.abc import MetaPathFinder
class NoTraining(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'sklearn'):
            raise AssertionError('Data preparation imported training dependency: ' + fullname)
sys.meta_path.insert(0, NoTraining())
from fraud_detector.dataset import build_training_features, validate_and_sort
import pandas as pd
raw = pd.DataFrame([dict(TRANSACTION_ID='1', TX_DATETIME='2024-01-01T12:00:00Z',
                        CUSTOMER_ID='c', TERMINAL_ID='t', TX_AMOUNT=10,
                        TX_FRAUD=0, TX_FRAUD_SCENARIO=0)])
frame = build_training_features(validate_and_sort(raw))
assert frame.iloc[0].amount == 10 and frame.iloc[0].customer_feedback_count == 0
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(Path("src").resolve())},
    )
    assert result.returncode == 0, result.stderr


def test_smoke_module_import_has_no_docker_or_argument_parsing_side_effects(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Import started an external process")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(sys, "argv", ["import-check", "--not-a-smoke-option"])
    module = importlib.import_module("scripts.smoke_compose")
    assert callable(module.main)


def test_legacy_entry_points_reexport_the_package_implementations():
    from apps.traffic_generator.main import post_transaction as replay_delivery
    from fraud_detector.dataset import prepare_dataset
    from fraud_detector.delivery import post_transaction
    from fraud_detector.journal import Stream
    from fraud_detector.model.training import train_model
    from fraud_detector.simulator import Stream as legacy_stream
    from scripts.prepare_dataset import prepare_dataset as legacy_prepare
    from scripts.train import train_model as legacy_train

    assert legacy_prepare is prepare_dataset
    assert legacy_train is train_model
    assert legacy_stream is Stream
    assert replay_delivery is post_transaction


def test_idempotency_retains_legacy_null_source_and_numeric_inputs_exclude_identifiers(transaction):
    transaction.pop("source_transaction_id")
    event = Transaction.model_validate(transaction)
    payload = event.idempotency_payload()
    assert payload["source_transaction_id"] is None
    assert "behavioral_features" not in payload and "adaptive_features" not in payload
    assert json.loads(json.dumps(payload))["transaction_id"] == transaction["transaction_id"]
    assert event.feature_values() == {
        "amount": 10.0,
        "transactions_last_hour": 0,
        "customer_history_days": 0.0,
        "hour_of_day": 12,
    }
