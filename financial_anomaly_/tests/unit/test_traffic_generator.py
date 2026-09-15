import json
from pathlib import Path

from apps.traffic_generator.main import replay_transactions


def test_replay_prefixes_ids_and_preserves_source_mapping(tmp_path: Path) -> None:
    replay_path = tmp_path / "replay.jsonl"
    replay_path.write_text(
        json.dumps({"transaction_id": "123", "amount": 10}) + "\n",
        encoding="utf-8",
    )

    events = replay_transactions(replay_path, "run001")

    assert events == [
        {
            "transaction_id": "run001:123",
            "source_transaction_id": "123",
            "amount": 10,
        }
    ]
