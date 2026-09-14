# Realtime Fraud Detector

An end-to-end realtime financial fraud detection platform built with PyTorch,
Docker, Terraform, Prometheus, and Grafana.

## Five-container runtime

Terraform provisions exactly five containers:

1. `traffic-generator` continuously posts synthetic transactions.
2. `api` validates events, scales features, runs the PyTorch scorer, exposes
   Prometheus metrics, and sends every prediction to the audit service.
3. `audit-store` persists predictions in a SQLite database.
4. `prometheus` scrapes the API metrics endpoint.
5. `grafana` displays provisioned live prediction, fraud-rate, and latency
   charts.

## Repository layout

```text
.
├── apps/
│   ├── audit_store/            # SQLite-backed prediction audit API
│   └── traffic_generator/      # Continuous synthetic transaction producer
├── configs/                    # Shared application and model configuration
├── data/
│   ├── raw/                    # Source datasets (not committed)
│   ├── interim/                # Validated/intermediate datasets
│   └── processed/              # Training-ready datasets
├── deploy/
│   ├── docker/                 # Service Dockerfiles
│   └── kubernetes/             # Optional Kubernetes manifests
├── docs/                       # Architecture and operational documentation
├── infra/terraform/
│   ├── environments/           # dev/staging/prod root modules
│   └── modules/                # Reusable cloud infrastructure modules
├── monitoring/                 # Prometheus and Grafana configuration
├── notebooks/                  # Exploratory analysis and experiments
├── scripts/                    # Data, training, and local operations scripts
├── src/fraud_detector/         # Shared Python package
├── tests/                      # Unit, integration, and contract tests
├── docker-compose.yml          # Local five-container development stack
├── pyproject.toml              # Python tooling and dependencies
└── .env.example                # Local configuration template
```

## Getting started

1. Copy `.env.example` to `.env` and adjust local values.
2. Start the five-container local stack with `docker compose up --build`.
3. Open the API at `http://localhost:8000/docs`, Prometheus at
   `http://localhost:9090`, and Grafana at `http://localhost:3000`.
4. To provision the same five-container topology with Terraform:
   `cd infra/terraform/environments/dev && terraform init && terraform apply`.

Terraform resolves the project root to an absolute path for Docker bind mounts.
The root `.dockerignore` excludes local virtual environments, Terraform state,
and generated caches so Docker image builds remain small and reliable.
Grafana mounts only its provisioning and dashboard directories, preserving the
image's built-in configuration and startup paths.

The Compose project name and locally built image tags are explicitly pinned, so
the stack can also be run from directories whose names contain underscores or
trailing separators.

The initial tree is intentionally scaffolded so model architecture, event
transport, persistence, and cloud-provider-specific Terraform can be added
without moving public interfaces.
