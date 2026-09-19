# Deployment status

**Not approved for production release.** The implementation targets highly available Kubernetes
with PostgreSQL; Compose/Terraform are local development paths.

Completed controls include authenticated ingestion/audit/metrics, validated inputs, bounded
requests, persistent idempotency across replicas, readiness checks, TLS configuration, model
release gates, replicated deployment manifests and backup configuration.

Release blockers:

- The demo artifact is the evaluated 18-feature gradient-boosted model. It is not production-approved.
  Production correctly rejects it. A new 90-day synthetic dataset now has 5,817 fraud cases;
  its first candidate also failed evaluation and was not promoted. See `docs/SIMULATOR.md`.
  Representative data and a reviewed passing policy remain necessary for real deployment.
- The image security gate fails. See `reports/security/summary.json`; upstream OS and bundled Go
  package findings remain. Update/rebuild affected images and rerun the gate for every deployed
  digest, including cluster operators and gateway. No vulnerability waivers are applied.
- No target cluster has been supplied or deployed. Server-side CRD validation, three-zone
  failover, off-cluster restore, load/SLO acceptance and alert delivery remain unverified.

Use [HA deployment](docs/HA_DEPLOYMENT.md) and [operations](docs/PRODUCTION_RUNBOOK.md).
Passing application tests alone does not establish model quality or operational availability.
