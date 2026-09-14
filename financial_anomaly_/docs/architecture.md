# Architecture

The platform is organized around an online transaction path and an offline
model lifecycle:

1. Transactions enter through the API or Kafka topic.
2. The stream processor validates events, computes online features, and calls
   the PyTorch inference component.
3. Scores and decisions are emitted for downstream consumers and metrics are
   exposed to Prometheus.
4. Training jobs consume versioned data from `data/`, evaluate a candidate
   model, and publish artifacts under `models/`.
5. Terraform provisions environment-specific infrastructure; Docker Compose
   provides the local equivalent for development.
