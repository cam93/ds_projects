# Full Handbook pipeline

Source: Fraud-Detection-Handbook/simulated-data-raw, commit
`6e67dbd0a3bfe0d7ec33abc4bce5f37cd4ff0d6a`, 183 daily files.

Raw files are preserved. Their Git blob hashes were checked against the pinned
GitHub tree before generating `configs/trusted-handbook-6e67dbd.json`.
Preparation verifies that SHA256 manifest before loading each pickle.

Cleanup quarantines 42 zero-amount, nonfraudulent transactions, which do not satisfy
the API's positive-amount contract. The remaining 1,754,113 transactions undergo
normal schema, amount, timestamp and unique-ID validation. No duplicates were found.

Outputs are in `data/processed/handbook-full`: quarantined records, clean records,
training features, all replay events and `handbook/replay.jsonl` containing only the
chronological held-out test partition with corresponding labels.

`run_pipeline.py` records the exact cleanup/training procedure used for this run.
It trains the existing 18-feature histogram gradient boosting configuration, uses
seven-day label-availability gaps before validation and test, and selects the
highest-precision validation operating point with recall above 85%. No FPR cap is
applied. The model and selection are saved before test evaluation. Candidate output
is unapproved and does not replace the active model.

`pipeline.log` contains progress, `cleanup.json` records row counts,
`selection-before-test.json` records selection and splits, and `training.json`
contains final test metrics. The candidate is `candidate.json`.

Replay uses the currently deployed API model. Its metrics are distinct from the
new candidate's offline test results. A dedicated replay run ID prevents collisions
with existing simulator and replay transactions. Ingestion checkpoints advance only
after the API acknowledges a transaction.

## Completed training results

- recall: 82.10%
- false_negative_rate: 17.90%
- precision: 1.02%
- false_positive_rate: 71.28%
- alert_rate: 71.38%

Test population: 263,045 transactions, including 2,330 fraud cases. The candidate missed 417 fraud cases and generated 185,836 false alerts. It was not promoted. Portable export agrees with native predictions to maximum absolute error 1.11e-16.

## Running ingestion

Container: `fraud-handbook-replay`. Replays all 263,045 held-out events to the existing API, at up to 10 transactions per second. It takes at least 7.3 hours, plus request and checkpoint overhead. Training and validation rows are not replayed. The existing simulator remains running.

```sh
docker logs --tail 20 fraud-handbook-replay
cat data/processed/handbook-full/replay-state/checkpoint.json
# Resume after a stop or exhausted retries:
docker start fraud-handbook-replay
# Stop this replay:
docker stop fraud-handbook-replay
```

The checkpoint `completed` count is the number acknowledged by the API. A successful finite run logs `replay_complete completed=263045` and exits. This additional container is not managed by Terraform; stop/remove it before destroying the demo stack.
