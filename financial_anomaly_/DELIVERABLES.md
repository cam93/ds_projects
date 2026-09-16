> Production target: three-zone Kubernetes with PostgreSQL. Compose/SQLite and Terraform are
> development paths. See [HA deployment](docs/HA_DEPLOYMENT.md) for the current deployment contract.

# Deliverables

- Authenticated API, read/write-separated ledger access, request limits and private monitoring.
- Durable idempotency, bounded replay retries, persisted checkpoints and conflict detection.
- Validated ingestion, held-out replay, minibatch training and explicit model release gates.
- Hash-pinned artifacts, constrained dependencies and non-root/read-only application containers.
- Unit/HTTP integration tests, isolated Docker smoke test and vulnerability scanning workflow.
- Backup/evaluation tools, deployment configuration and an operational release runbook.

These deliverables do not certify detection quality or operational readiness. See
DEPLOYMENT_READY.md for the current release status.
