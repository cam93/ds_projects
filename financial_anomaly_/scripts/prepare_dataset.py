"""Compatibility entry point for dataset validation and preparation."""

from fraud_detector.dataset import (
    REQUIRED_COLUMNS,
    build_online_features,
    build_replay_events,
    build_training_features,
    main,
    parse_args,
    prepare_dataset,
    prepare_pickles,
    read_source,
    validate_and_sort,
    write_outputs,
)

__all__ = [
    "REQUIRED_COLUMNS",
    "build_online_features",
    "build_replay_events",
    "build_training_features",
    "main",
    "parse_args",
    "prepare_dataset",
    "prepare_pickles",
    "read_source",
    "validate_and_sort",
    "write_outputs",
]

if __name__ == "__main__":
    main()
