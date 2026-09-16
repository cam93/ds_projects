# Deployment status

**Not approved for production release.** The implementation targets highly available Kubernetes
with PostgreSQL; Compose/Terraform are local development paths.

Completed controls include authenticated ingestion/audit/metrics, validated inputs, bounded
requests, persistent idempotency across replicas, readiness checks, TLS configuration, model
release gates, replicated deployment manifests and backup configuration.

Release blockers:

- The supplied artifact has three training fraud cases and no validation/test fraud cases.
  Production correctly rejects it. Obtain representative labeled data and pass a reviewed policy.
- The image security gate fails. See `reports/security/summary.json`; upstream OS and bundled Go
  package findings remain. Update/rebuild affected images and rerun the gate for every deployed
  digest, including cluster operators and gateway. No vulnerability waivers are applied.
- No target cluster has been supplied or deployed. Server-side CRD validation, three-zone
  failover, off-cluster restore, load/SLO acceptance and alert delivery remain unverified.

Use [HA deployment](docs/HA_DEPLOYMENT.md) and [operations](docs/PRODUCTION_RUNBOOK.md).
Passing application tests alone does not establish model quality or operational availability.
