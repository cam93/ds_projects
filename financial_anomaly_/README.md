# Realtime Fraud Detector

An end-to-end realtime financial fraud detection platform built with PyTorch,
Docker, Terraform, and Prometheus.

## Repository layout

```text
.
├── apps/
│   ├── api/                    # HTTP inference and health endpoints
│   └── stream_processor/       # Transaction ingestion and online scoring
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
├── docker-compose.yml          # Local end-to-end development stack
├── pyproject.toml              # Python tooling and dependencies
└── .env.example                # Local configuration template
```

## Getting started

1. Copy `.env.example` to `.env` and adjust local values.
2. Start the local dependencies with `docker compose up --build`.
3. Run the API locally with `uvicorn fraud_detector.api:app --reload`.
4. Train a baseline model with `python scripts/train.py`.

The initial tree is intentionally scaffolded so model architecture, event
transport, persistence, and cloud-provider-specific Terraform can be added
without moving public interfaces.
