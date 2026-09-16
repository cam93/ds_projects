> Production target: three-zone Kubernetes with PostgreSQL. Compose/SQLite and Terraform are
> development paths. See [HA deployment](HA_DEPLOYMENT.md) for the current deployment contract.

# Architecture

Trusted labeled data → validated chronological features → gated candidate training → approved
hash-pinned model. The test partition alone produces replay events and evaluation labels.

Authenticated enriched events → API validation → idempotency lookup → model scoring → atomic
ledger insert → persisted response. A duplicate input returns its original decision; conflicting
input under the same ID returns 409. Persistence failures return 503. No decision is acknowledged
before ledger commit. PostgreSQL unique keys arbitrate concurrent writes across ledger replicas. Synchronous replication
protects acknowledged commits against a single database-node failure; SQLite WAL is development-only.

The API does not accept raw customer events or calculate online history. A trusted feature producer
must supply the historical fields. The offline Handbook implementation is shared and tested.

Compose runs private API, ledger and monitoring services. Replay and the external TLS gateway are
optional profiles. Terraform targets an existing local Docker host and provisions private services;
it does not create cloud infrastructure. See PRODUCTION_RUNBOOK.md for the deployment boundary.
