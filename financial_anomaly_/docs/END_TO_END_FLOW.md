> Production target: three-zone Kubernetes with PostgreSQL. Compose/SQLite and Terraform are
> development paths. See [HA deployment](HA_DEPLOYMENT.md) for the current deployment contract.

# End-to-end flow

Current setup, data preparation, training, deployment and verification commands are maintained in
[README.md](../README.md). Operational recovery and acceptance requirements are in
[PRODUCTION_RUNBOOK.md](PRODUCTION_RUNBOOK.md).

Preparation writes all training features plus held-out-only replay events and matching labels.
Training uses chronological partitions with tied timestamps kept together. Release gates require
adequate class counts and policy metrics. API startup verifies approval and artifact integrity.
Replay checkpoints progress; the API and ledger make retries idempotent. Evaluate a complete replay
against its held-out labels using scripts/evaluate_replay.py and a consistent PostgreSQL snapshot (or a local SQLite backup).
