"""Backward-compatible serialization of point-in-time feature history."""

from collections import deque
from dataclasses import asdict
from datetime import datetime

from fraud_detector.features.handbook import FeatureState, RunningStats


def encode_features(state):
    return {
        "customer_stats": {key: asdict(value) for key, value in state.customer_stats.items()},
        "terminal_stats": {key: asdict(value) for key, value in state.terminal_stats.items()},
        "customer_terminals": state.customer_terminals,
        "terminal_recent": {
            key: [stamp.isoformat() for stamp in stamps]
            for key, stamps in state.terminal_recent.items()
        },
        "latest_timestamp": state.latest_timestamp.isoformat() if state.latest_timestamp else None,
        "first": {key: value.isoformat() for key, value in state.first_seen.items()},
        "recent": {
            key: [value.isoformat() for value in values]
            for key, values in state.recent_transactions.items()
        },
    }


def decode_features(value):
    return FeatureState(
        customer_stats={
            key: RunningStats(**stats) for key, stats in value.get("customer_stats", {}).items()
        },
        terminal_stats={
            key: RunningStats(**stats) for key, stats in value.get("terminal_stats", {}).items()
        },
        customer_terminals=value.get("customer_terminals", {}),
        terminal_recent={
            key: deque(datetime.fromisoformat(stamp) for stamp in stamps)
            for key, stamps in value.get("terminal_recent", {}).items()
        },
        latest_timestamp=datetime.fromisoformat(value["latest_timestamp"])
        if value.get("latest_timestamp")
        else None,
        first_seen={key: datetime.fromisoformat(stamp) for key, stamp in value["first"].items()},
        recent_transactions={
            key: deque(datetime.fromisoformat(stamp) for stamp in stamps)
            for key, stamps in value["recent"].items()
        },
    )
