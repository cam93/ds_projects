# Fraud transaction scoring service

An authenticated transaction-scoring API, PostgreSQL production ledger (SQLite for development), offline training pipeline,
restartable dataset replay, and private Prometheus/Grafana monitoring.

**Release status: hardened implementation; model and operational acceptance still required.**
The supplied one-day model is deliberately rejected by production startup. Its training split
has three fraud examples and validation/test have none. Passing software tests does not establish
fraud-detection quality on real transactions.

## Supported deployment

The production target is a three-zone Kubernetes deployment with replicated API/ledger services,
synchronous PostgreSQL replication, TLS between services and off-cluster backups. See
[the HA deployment guide](docs/HA_DEPLOYMENT.md) for prerequisites and staged rollout.
Compose and Terraform remain single-host development tools. Producers must be trusted and
authenticated, sending enriched transactions.
The API credential belongs to the trusted feature pipeline, never to a browser/mobile client.
The service verifies UTC hour consistency and input bounds; it cannot verify historical counts
without access to the producer's underlying transaction history. The offline Handbook preparer
computes these features from ordered records. Integrate an authoritative online feature service
before accepting raw live customer transactions.

## Install and test

Use Python 3.10 and a virtual environment. From this project directory:

```bash
python -m pip install --no-deps --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
python -m pip install -c constraints.txt -e '.[dev,data]'
python -m pytest -q
python -m ruff check src apps scripts tests
```

On macOS, install PyTorch from the standard Python package index instead of the Linux CPU index.
The constraints file pins tested packages. Docker pins its Python base by digest and uses CPU
PyTorch. CI builds the image, runs isolated HTTP tests, and rejects HIGH/CRITICAL image findings.

## Prepare data

Prefer CSV with the Handbook columns or Parquet (install pyarrow separately for Parquet).
Pickle files require a reviewed manifest mapping each filename to its SHA256. A checksum does
not make an unknown pickle safe: obtain files from the official repository and verify provenance
before recording them in the manifest. Deserialization occurs only after the hash matches.

```bash
python scripts/prepare_dataset.py data/raw/transactions.csv --output-dir data/processed
# Verified Handbook pickle input:
python scripts/prepare_dataset.py data/raw/*.pkl --trusted-manifest configs/trusted-data.json
```

Preparation validates IDs, binary labels, finite amounts and API bounds. It calculates features
once across all days, writes all features for training, and writes only the chronological test
partition to `data/processed/handbook/replay.jsonl`, with matching `test_labels.csv`.
The hourly window includes preceding events exactly one hour earlier. Ties follow deterministic
transaction ID order; the current event is excluded. Late events are rejected by the feature
calculator. Source timestamps lacking a timezone are interpreted as UTC by the Handbook importer.

## Train and approve an artifact

Copy `configs/release-policy.example.json` to a reviewed policy. Its thresholds are examples,
not validated business requirements. Choose acceptable precision/recall and required fraud counts
for your application. Obtain enough labeled days for all chronological partitions.

```bash
python scripts/train.py --input data/processed/training_features.csv \
  --output models/artifacts/candidate.pt --policy configs/release-policy.json
```

Training rejects insufficient/single-class splits, fits normalization on training only, uses
minibatches, selects the threshold on validation, and gates export on test metrics. Equal timestamps
stay in the same partition. Reports include class counts, source/policy hashes and a unique version.
The output is a model, report, and SHA256 sidecar. Use new output filenames for each candidate;
review the report before copying an accepted candidate to `models/artifacts/fraud_model.pt`.
Do not repeatedly tune against the same test set. Synthetic evaluation does not establish
performance on a different real-world population; require a representative shadow evaluation.
The reported sigmoid is a model score, not a demonstrated calibrated probability.

## Configure and start the local stack

```bash
python scripts/init_secrets.py
cp .env.example .env
```

Set `MODEL_SHA256` in `.env` to the approved artifact's SHA256. The API verifies both this hash and
its embedded release approval at startup. Secret files stay outside Git/images. Distinct bearer
keys protect prediction ingestion, audit writes, audit reads and metrics. Grafana allows anonymous **Viewer** access for this synthetic-data demo and opens the fraud
dashboard by default. Visitors can view charts but cannot save dashboards, configure data sources
or administer Grafana. Administrator sign-in remains available at `/login` using its separate
password. To rotate keys, replace the files and restart all affected services together.

```bash
docker compose up --build -d
# Explicit finite replay; progress survives container recreation:
docker compose --profile replay up --build traffic-generator
```

API, Grafana, and Prometheus host ports bind only to loopback. The audit service has no host port.
All application containers run as UID 10001 with read-only roots, dropped capabilities and resource
limits. The internal backend network carries service traffic. Secrets files are read-only inside
containers; restrict host directory access and encrypt the host disk/backups where required.

To deliberately run a new replay, use a new checkpoint path or a new replay-state volume.
Do not erase the current checkpoint to recover a transient failure. The same request ID and body
return the original persisted result, even across model changes. Reusing an ID with different
input returns 409. Permanent errors stop replay; transient failures get six bounded attempts.

For an external trusted producer, set `PUBLIC_HOST` to your DNS name and use
`docker compose --profile public up -d`. The optional Caddy gateway provisions TLS and exposes
only `/predict`; configure DNS/firewall first. Never expose audit/metrics/Grafana directly.
Terraform provisions the private single-host **development** stack from the project directory:

```bash
# First checkout only (already initialized in this workspace):
terraform init
# Normal lifecycle:
terraform apply
terraform destroy
```

The model SHA256 is calculated automatically from `models/artifacts/fraud_model.pt`. No shell
hash command, `-chdir`, `-var`, or `-parallelism=1` is needed. The project-root entry point uses
`infra/terraform/environments/dev/terraform.tfstate`, preserving the original state and resource
addresses. Run only one Terraform operation at a time; prefer this entry point going forward.
On a fresh checkout, generate secrets with `python scripts/init_secrets.py` first and ensure the
local model exists. Development mode permits the bundled unapproved model, while still checking
its hash and keeping authentication enabled. This does not change production release gates.

Terraform builds one shared application image using Docker's `default` BuildKit builder; the
first uncached build downloads PyTorch and may take several minutes. Progress is available in
`terraform-build.log` (`tail -f terraform-build.log`). Source changes, dependency constraints and
Dockerfile changes trigger rebuilds; Python caches do not. API/ledger creation waits for readiness
and fails after 120 seconds if unhealthy. Diagnose failures with `docker logs fraud-dev-api` and
`docker logs fraud-dev-audit-store`.

Replay is disabled by default; use `terraform apply -var='enable_replay=true'` to opt in. It is a
finite job, so a successful exit is expected. `terraform destroy` removes the managed volumes,
including local audit records and replay checkpoints; back up data you intend to retain.
Terraform does not create cloud machines or HA infrastructure. Do not run Compose and Terraform
against the same deployment. The production path is documented in `docs/HA_DEPLOYMENT.md`.

See [operations and release requirements](docs/PRODUCTION_RUNBOOK.md) for migration, backup,
retention, alerts, replay evaluation, rollout and remaining acceptance work.
