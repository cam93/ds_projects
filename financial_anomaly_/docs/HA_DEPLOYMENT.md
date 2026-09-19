# Highly available production deployment

This is a deployment candidate, not evidence of a running or accepted production cluster.
The renderer produces manifests without contacting a cluster. Review `DEPLOYMENT_READY.md`
for current model, vulnerability and operational release blockers.

## Platform prerequisites

Provide a Kubernetes cluster with at least three schedulable zones, adequate spare capacity for
rolling updates, encrypted persistent volumes, a default storage class, metrics-server, and a
CNI that enforces NetworkPolicy. Install compatible versions of Gateway API (v1 BackendTLSPolicy,
1.4 or later), Envoy Gateway, cert-manager, CloudNativePG (1.28 API), its Barman Cloud plugin,
Prometheus Operator and Stakater Reloader. Pin and scan their images too. Operator installation,
cloud networking, IAM and object-store provisioning are platform responsibilities.

The manifest assumes namespaces `envoy-gateway-system`, `cnpg-system`, `monitoring` and DNS in
`kube-system`; change network policies if your platform differs. PostgreSQL egress permits port
443 for the Kubernetes API and S3; adapt to your API endpoint port and restrict destinations
through the platform egress firewall. IPv6-only clusters need corresponding policy changes.
Configure encrypted S3 storage, least-privilege bucket credentials and backup access auditing.
Do not share the backup prefix with another active cluster.

## Build and render

Build the application runtime with `deploy/docker/api.Dockerfile`, scan it, publish to your
registry, and record the registry digest. Both API and ledger code are included. Build the API
release image using `deploy/docker/release.Dockerfile` and `--build-arg RUNTIME_IMAGE=...@sha256:...`;
its build context must contain the approved `models/artifacts/fraud_model.json`. Scan the final image.
The ledger uses the runtime digest. Select a CloudNativePG-compatible PostgreSQL image, not the
plain Docker PostgreSQL image used by the local integration test.

```bash
python scripts/render_kubernetes.py \
  --api-image "$API_RELEASE_DIGEST" --ledger-image "$RUNTIME_DIGEST" \
  --postgres-image "$CNPG_POSTGRES_DIGEST" --model-hash "$MODEL_SHA256" \
  --hostname fraud.example.com --internal-issuer private-ca --public-issuer public-ca \
  --gateway-class eg --backup-uri s3://YOUR-BUCKET/fraud \
  --output /tmp/fraud-release.json
```

The images and model hash must be real immutable digests. Certificate issuers must already exist;
the internal issuer must issue certificates trusted by the supplied internal CA. The API image
contains the model so every replica runs identical bytes. Production startup rejects an unapproved
model or a hash mismatch. Never use APP_ENV=development in this deployment.

## Secrets and staged rollout

Create namespace `fraud` first with restricted Pod Security labels. Supply secrets from your
secret manager, enable Kubernetes secret encryption and restrict RBAC. Do not commit credentials.
Required objects in `fraud`:

| Object | Keys / purpose |
| --- | --- |
| Secret `fraud-api-credentials` | `api_key`, `audit_write_key`, `metrics_key` |
| Secret `fraud-audit-store-credentials` | same `audit_write_key`, separate `audit_read_key` |
| ConfigMap `fraud-internal-ca` | `ca.crt`, internal certificate issuer trust chain |
| Secret `fraud-db-owner` | `username=fraud_owner`, `password`; type kubernetes.io/basic-auth |
| Secret `fraud-db-app` | `username=fraud_app`, `password`; type kubernetes.io/basic-auth |
| Secret `fraud-db-ca` | `ca.crt`, copied from the cluster's database CA after bootstrap |
| Secret `fraud-database` | `url`, runtime PostgreSQL URL for fraud_app |
| Secret `fraud-migration-database` | `url`, migration PostgreSQL URL for fraud_owner |
| Secret `fraud-backup-credentials` | `ACCESS_KEY_ID`, `ACCESS_SECRET_KEY` |

Generate distinct random bearer keys of at least 32 characters. PostgreSQL URLs use
`fraud-db-rw.fraud.svc.cluster.local:5432/fraud?sslmode=verify-full&sslrootcert=/run/db-ca/ca.crt`;
URL-encode usernames/passwords. Never put migration credentials in application pods. The migration
creates the table and grants only SELECT/INSERT to fraud_app. Database owner credentials remain
privileged and require separate access controls.

1. Validate against installed CRDs using `kubectl apply --dry-run=server -f /tmp/fraud-release.json`.
   This is required; structural unit tests do not validate platform compatibility.
2. Apply namespace/service account, certificates, network policies, ObjectStore and database
   Cluster first. Wait for all three database instances and their synchronous replication to be
   healthy. Copy the database CA into the required secret, and create the verified-TLS URL secrets.
3. Apply the schema migration Job and wait for successful completion. It is advisory-lock
   serialized and repeatable. A failed job or missing table must block application rollout.
   Delete/recreate the completed migration Job for a later migration; do not rely on changing an
   immutable Job template.
4. Apply Deployments, Services, PDBs, HPAs, ServiceMonitor, EnvoyProxy, Gateway and routes. Wait for
   certificates, all replicas and Gateway/HTTPRoute/BackendTLSPolicy acceptance conditions.
5. Configure Prometheus to select this ServiceMonitor and PrometheusRule, route alerts to an
   attended receiver, and verify authenticated TLS scraping. Configure database replication,
   disk usage, certificate expiry and backup-age alerts in the platform monitoring stack.
6. Enable the ScheduledBackup; verify a completed base backup and continuous WAL archival.
   Run the acceptance exercises below before routing real producers.

Readiness removes replicas when dependencies fail. Liveness checks only the process. The gateway
exposes POST /predict only. Ledger and metrics remain internal. HTTP clients must preserve the
transaction ID and normalized request body after timeouts: a lost response may follow a committed
write. A retry returns the original record, including its original model version.

## Availability, rotation and recovery

Three API and ledger replicas spread across zones. PostgreSQL requires one synchronous standby
before acknowledging a commit. Losing quorum deliberately stops writes; application retries get
503. This is not a guarantee against every correlated failure. The database primary's service
address stays stable during failover; pools discard broken connections and reconnect.

Reloader rolls pods after secret/certificate changes. Bearer-key rotation requires a coordinated
maintenance window because this implementation accepts one active key per role; do not promise
uninterrupted rotation. Update producers, API and ledger together. Rotate database passwords and
CA chains with the platform's documented staged procedure and verify all connections afterward.

Use CloudNativePG recovery into a **new cluster** from the Barman object store for disaster recovery.
Validate row counts, canary IDs, integrity and recovery point before changing service routing.
Daily backups plus continuous WAL support point-in-time recovery; the actual RPO/RTO depend on
successful archive and restore tests. Keep the old cluster isolated for diagnosis. No automatic
retention deletion is configured in the production ledger; define data retention and provision
capacity. Deleting an audit row also deletes its idempotency protection.

Before release, exercise a pod kill, worker-node drain, primary failover, complete zone loss,
network/database outage, lost response after commit, certificate rotation, and off-cluster restore.
Confirm acknowledged predictions survive, duplicate retries do not create new records, 503s are
bounded and alerts arrive. Load-test with representative enriched events and an agreed SLO.
The included isolated PostgreSQL test covers concurrency, role restrictions and restart recovery;
it does not substitute for these cluster exercises.

Rollback by redeploying the previous approved API image/hash. Do not roll back the audit database
or erase predictions. Schema changes must remain compatible with both versions during rolling
updates. Model quality, calibration, privacy/retention and producer feature integrity require
explicit operational acceptance before live fraud decisions.
