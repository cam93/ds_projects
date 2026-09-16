# Operations and release requirements

## Release boundary

The supplied data/model are development artifacts. Production boot requires a pinned SHA256 and
an artifact produced by passing the reviewed release policy. No bypass is configured in Compose
or Terraform. `APP_ENV=development` exists only for controlled local tests; it does not disable auth.

Before a real release, approve these deployment-specific items:

- Representative labeled data, adequate fraud counts and accepted precision/recall; shadow-test
  real traffic and probability calibration if scores are used as probabilities.
- A trusted feature producer using the same definitions; raw browser-supplied historical features
  are not trustworthy. Current support is authenticated enriched events.
- Throughput/SLO, host disk capacity, retention period, backup schedule and tested restore time.
- Alert delivery destination/on-call owner, public DNS/TLS if needed, secret ownership/rotation.
- Current image scan results, reviewed dependency/base-image updates, and a rollback artifact.

Production uses the replicated PostgreSQL topology in [HA deployment](HA_DEPLOYMENT.md).
SQLite is permitted only with APP_ENV=development. Synchronous commits require an available
standby; loss of quorum stops writes rather than acknowledging unreplicated predictions.
Failover and disaster recovery still require testing in the target cluster.

## Existing local SQLite deployment migration

1. Stop replay and route producers away from the old API.
2. Back up the old SQLite database with the backup command below. Do not copy only the main
   `.db` file while a WAL-mode writer is active.
3. Ensure the audit volume is writable by UID/GID 10001. Older root-owned volumes require an
   operator-controlled ownership migration on the exact volume after backup.
4. Generate credentials, prepare an approved model/hash, and deploy. Startup adds missing audit
   columns and enables WAL/full synchronous commits. Existing legacy rows remain intact.
5. Legacy records lack request hashes and cannot safely be replayed with their original IDs;
   they return 409. Use a new run namespace. New rows support authenticated cursor pagination.
6. Check `/ready`, send a canary, and verify persistence before restoring producer traffic.

## Local SQLite backup, restore and retention

Use `python scripts/maintain_audit.py /path/audit.db /path/backups/audit-YYYYMMDD.db` from a
maintenance environment with the audit volume mounted. It uses SQLite's backup API and checks
integrity. Store the result off-host with encryption/access control and a defined retention policy.
The backup destination must not already exist.

To prune after a verified backup, add `--prune-before 2026-01-01T00:00:00Z` with your approved
UTC cutoff. Pruning uses ingestion time, not the historical transaction timestamp. Legacy rows
without ingestion times are retained. Deleting rows also removes their idempotency protection;
never retry transactions older than the retention horizon. VACUUM is not automatic; schedule
space reclamation during maintenance if needed.

To restore: stop writers, retain the damaged database for diagnosis, restore the verified backup
to a clean audit volume, set UID/GID 10001 ownership, start the ledger, check integrity/readiness,
then resume the API and producers. Restore into a separate test volume periodically.

## Monitoring

Prometheus scrapes authenticated metrics. Alert rules cover an unavailable API, persistence
failures and p95 latency above one second. These are rules only: configure an Alertmanager or
managed notification receiver before unattended deployment. Grafana and Prometheus are private
loopback interfaces. Add host disk-space and backup-age monitoring in the deployment platform.

`fraud_requests_total` counts authenticated request outcomes, including idempotent replays;
`fraud_predictions_total` counts newly inserted predictions. Request latency includes failures.
Inspect process logs for failure/progress outcomes; do not log credentials or complete transaction
payloads. Track real labeled performance separately from predicted fraud rate. Retrain/review on
measurable drift; do not automatically replace a model based solely on a distribution change.

Defaults: 16 KiB request bodies, ten-second body read limit, 32 concurrent admitted requests,
1,200 requests/minute token bucket per process (initial burst up to 1,200), 64 Uvicorn concurrency
limit. One worker is intentional. Each new prediction makes two ledger calls, so the ledger's
rate limit also bounds total throughput. These limits are per replica, not a global tenant quota. Reassess them through load tests and
configure producer quotas at the ingress platform.

## Replay and evaluation

Replay is an explicit Compose profile, never automatic production traffic. Its dataset checksum,
run ID and completed-record count persist in `/state/checkpoint.json`. A changed dataset refuses
the old checkpoint. After a failure, fix the cause and rerun the same checkpoint. Six retries cover
transient HTTP/transport failures; invalid input/auth/conflicting IDs stop immediately.

Evaluate one complete run against held-out labels from a consistent ledger backup:

```bash
python scripts/evaluate_replay.py backups/audit.db data/processed/handbook/test_labels.csv \
  --run-id YOUR_REPLAY_RUN_ID
```

The evaluator rejects missing/extra/duplicate joins, mixed model versions and single-class labels.
It uses persisted decisions and original scores; it never silently drops unmatched labels.

For production evaluation, use a dedicated SELECT-only database role and a verified-TLS URL
file: add `--postgres-url-file /secure/evaluation-url` to the command above (the database
positional argument is ignored). The query uses a read-only repeatable-read snapshot.

## Rollback and artifact policy

Keep approved versioned model files and their reports/hashes. To roll back, install the previous
artifact at the configured path, set its pinned hash, and recreate the API. Existing transaction
IDs still return original decisions. Use new IDs only for explicitly new transactions/replays.
A missing, mismatched, malformed or unapproved model fails startup instead of silently falling
back to a baseline. Model changes need the same offline and shadow acceptance as initial release.

## Verification

Run unit/integration tests, lint and formatting checks. Build the API image, then run
`python scripts/smoke_containers.py` with Docker available. The smoke test creates isolated
containers/network and test-only credentials/model, verifies real HTTP calls and concurrent
idempotency, and removes its containers afterward. It never approves the shipped model.
Run `python scripts/smoke_postgres.py` for real PostgreSQL concurrency, restricted-role,
evaluation and restart checks, and `python scripts/smoke_compose.py` for the full local stack.
Both create temporary isolated resources and remove them afterward.
CI additionally scans the image for HIGH/CRITICAL package vulnerabilities. Pin/update images
through reviewed changes and rerun scans; a clean historical scan is not a permanent guarantee.
