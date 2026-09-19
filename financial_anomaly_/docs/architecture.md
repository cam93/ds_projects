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

## Code ownership after the refactor

| Responsibility | Implementation |
| --- | --- |
| Dataset loading, validation and materialization | `src/fraud_detector/dataset.py` |
| Shared batch/live enrichment and request construction | `src/fraud_detector/features/pipeline.py` |
| Backward-compatible feature checkpoint serialization | `src/fraud_detector/features/checkpoint.py` |
| Pending simulator requests, state migration and acknowledgements | `src/fraud_detector/journal.py` |
| Batch generation, stream orchestration and simulator CLI | `src/fraud_detector/simulator.py` |
| Bounded HTTP retries shared by replay and simulation | `src/fraud_detector/delivery.py` |
| Chronological splits, classification metrics, thresholds and intervals | `src/fraud_detector/model/evaluation.py` |
| Neural-network fitting and legacy release-gated training | `src/fraud_detector/model/training.py` |
| Shared candidate fitting, normalization and native/portable parity | `src/fraud_detector/model/candidates.py` |
| Data-only export of fitted model parameters | `src/fraud_detector/model/export.py` |
| Dataset/model checksums | `src/fraud_detector/artifacts.py` |

The existing `scripts/train.py` and `scripts/prepare_dataset.py` commands remain compatibility entry
points. They also re-export their existing public functions. Comparison and rolling-evaluation
scripts own their different selection policies and reports; both use the same fitting and scoring
implementation. Changing a candidate's training configuration now has one implementation to review.
The legacy F1 threshold policy and the two comparison policies remain distinct and unchanged.

Runtime services do not import executable CLI scripts or offline-training helpers; PyTorch remains
an inference dependency. Replay and simulation both depend on the shared delivery helper. Dataset preparation needs the data dependencies but does
not import PyTorch or scikit-learn. Candidate fitting imports those optional training dependencies
only when the training/comparison modules are used.

`Transaction` owns numeric input extraction and the legacy-compatible idempotency payload. Historical
request hashes retain `source_transaction_id: null` while omitting absent feature blocks. The journal
keeps its existing schema, serialized feature keys and exact retry bodies, so existing checkpoints
need no new migration for this refactor. All artifact formats, release gates and production
configuration are preserved. Importing the Compose smoke module no longer launches containers.

## Refactor verification

Before/after deterministic checks compared prepared files and all six candidates' prediction files,
thresholds, metrics and exported artifacts. The rolling check used the existing policy with two
training epochs for each version of the code; all 108 window/threshold combinations, final predictions
and exported artifacts matched exactly. These runs verify equivalence, not improved model quality,
and their temporary artifacts do not replace the existing model or evaluation reports.

The test suite includes compatibility imports, optional-dependency isolation, side-effect-free
smoke-module imports and legacy request serialization. Container checks cover legacy PyTorch and
portable JSON scoring, HTTP authentication, durable/idempotent ledger writes, simulator restart,
quality metrics, anonymous read-only Grafana and the PostgreSQL ledger.
