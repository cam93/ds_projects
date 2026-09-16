# Synthetic transaction simulator

The simulator creates new transactions rather than replaying a fixed file. It matches the nine
Handbook source columns: TRANSACTION_ID, TX_DATETIME, CUSTOMER_ID, TERMINAL_ID, TX_AMOUNT,
TX_TIME_SECONDS, TX_TIME_DAYS, TX_FRAUD and TX_FRAUD_SCENARIO. CSV stores UTC timestamps in ISO
format; ingestion converts IDs/timestamps to its normal types. This is our seeded simulator,
not an exact reproduction of the Handbook algorithm or its real-world distribution.

Customers have persistent spending amounts, activity rates, preferred shopping hours and familiar
terminals. Weekends change activity. Fraud includes unusual purchases (scenario 1), weekly groups
of compromised terminals (2), and spending bursts from compromised customers (3). Legitimate
large purchases overlap fraudulent amounts. Compromise groups and individual outcomes are random;
configured compromise fractions are not guarantees of a particular realized fraud percentage.
Set all three fraud rates to zero to generate an all-legitimate dataset.

## Start the live demo

From the project root, after the usual initial secret setup:

```bash
terraform apply
```

The local Terraform stack now starts `fraud-dev-simulator` after the API is healthy. It sends up
to four transactions per second using seed 17. Open http://localhost:3000 to see activity; allow
about 30 seconds for the first Prometheus scrapes and rate charts. For
service logs use `docker logs -f fraud-dev-simulator`. To stop synthetic traffic through Terraform,
use `terraform apply -var='enable_simulator=false'`. Explicit `enable_replay=true` takes precedence
and disables the simulator so a labeled replay is not mixed with generated traffic.

The simulator has its own persistent Docker volume. Container restarts preserve customer history,
run identity, event position and any pending request. `terraform destroy` removes this volume too.
A new volume creates a new run ID. Containers retry failures at most three restarts after the
HTTP client's bounded retries; inspect the error before restarting a stopped producer.

For a standalone producer against an already running API:

```bash
.venv/bin/python scripts/simulate.py stream --seed 17 --rate 4
# Deliver a finite additional set; otherwise stream runs until Ctrl+C:
.venv/bin/python scripts/simulate.py stream --seed 17 --count 100 --rate 4
```

Default endpoint: http://localhost:8000. Default credential: secrets/api_key. Default state:
data/simulator/live.db. Override using --url, --api-key-file and --state. Do not run a standalone
producer alongside the Terraform producer unless you deliberately want two independent runs.
Credentials are never written into checkpoints or batch manifests.

The delivery rate is wall-clock pacing; transaction timestamps follow an **accelerated simulated
clock**, starting at midnight UTC on the first stream invocation. This preserves daily patterns
and historical-feature definitions when the dashboard is sped up. Dates may move ahead of actual
calendar time. Grafana's operational metrics reflect ingestion time. This is live generated
traffic, not a simulation of real-time event latency. Batch and stream use identical feature
logic. --start accepts a UTC date or an explicit midnight UTC timestamp and is retained on resume.
Rate may change on restart; seed, population, fraud configuration, starting date and API target
must match the saved state. To change those, use a new state path/volume instead of deleting
pending requests. Do not reuse simulator state against a destroyed/recreated empty audit ledger
when assessing historical completeness; start a fresh run.

Compose users can opt in with `docker compose --profile simulation up --build -d`. Its existing
API artifact/environment requirements still apply; Terraform is the default demo path.

## Expand training data

Install the project's data dependencies first (see README). The generated 90-day dataset is
already available locally at `data/raw/simulated-demo.csv`; prepared outputs are under
`data/processed/simulated-demo`. Files are excluded from Git. The reproducible summary is
`reports/simulation/dataset-summary.json`.

To create another dataset without overwriting the first:

```bash
.venv/bin/python scripts/simulate.py batch --days 90 --start 2024-01-01 \
  --config configs/simulation.demo.json --output data/raw/simulated-next.csv
.venv/bin/python scripts/prepare_dataset.py data/raw/simulated-next.csv \
  --output-dir data/processed/simulated-next
```

Batch generation holds one day in memory and writes a checksum/configuration/count manifest next
to the dataset. It refuses existing output paths. CSV requires no additional writer dependencies.
Use a `.parquet` output filename if pyarrow is installed. No pickle deserialization is required.
Same seed, config, start date and generator/Python version reproduce the same CSV content.

The configuration file controls population, mean daily activity per customer and fraud scenarios.
An optional `drift_day` changes spending from that day onward by `drift_multiplier` (including
legitimate customers). Keep training/validation/test chronological. Use a separate seed to check
generalization to another population and do not tune against the test partition.

```bash
.venv/bin/python scripts/train.py \
  --input data/processed/simulated-demo/training_features.csv \
  --output models/artifacts/your-new-candidate.pt \
  --policy configs/release-policy.example.json
```

The existing policy is illustrative. Sufficient fraud examples do not guarantee its performance
thresholds will pass. In particular, the current four model features omit terminal history, so
terminal-compromise fraud intentionally includes cases the current model cannot distinguish.
Do not make labels trivially predictable or weaken a policy solely to obtain a passing score.
Training saves a report and refuses model export when acceptance fails. No model is automatically
promoted by data generation or live simulation.

## Ground truth and crash recovery

TX_FRAUD and TX_FRAUD_SCENARIO stay in the local simulator journal, never in scoring requests.
Only validated API fields cross the ingestion boundary. SQLite commits the enriched pending
request and updated feature history before HTTP submission; after acknowledgement it records the
response. If the response/checkpoint is lost after the API commits, the same ID and body are
retried, letting the API return its original durable decision. A process lock prevents concurrent
writers using the same state file. Do not share a state file over a network filesystem.

Export acknowledged truth/prediction pairs for analysis:

```bash
.venv/bin/python scripts/simulate.py export-labels \
  --state data/simulator/live.db --output data/simulator/evaluation.csv
```

For the Terraform producer, export inside its container and copy the result:

```bash
docker exec fraud-dev-simulator python -m fraud_detector.simulator export-labels \
  --state /state/live.db --output /state/evaluation.csv
docker cp fraud-dev-simulator:/state/evaluation.csv ./simulation-evaluation.csv
```

Choose a new output name for subsequent exports. Exports include source ID, synthetic label,
scenario, predicted decision, score and model version; they contain acknowledged events only.
Group by model version for evaluation. The journal grows with delivered transactions; stop the
demo when unused, archive required results and start a new run to reclaim space. Do not delete
individual journal/checkpoint rows to reset progress.

## First expanded-data evaluation

The initial 30-epoch candidate report is `models/artifacts/simulated-candidate.json`. On its
39,191-row test split (500 fraud cases), precision was 0.301, recall 0.344 and average precision
0.256. It failed the example 0.8 acceptance thresholds, so no candidate weights were exported
and the deployed artifact was not replaced. The dataset now supports meaningful evaluation;
the subsequent [behavioral-feature comparison](MODEL_COMPARISON.md) evaluates that improvement.
These scores concern synthetic data.

The simulator now sends a complete `behavioral_features` block using the same implementation as
offline preparation. On first resume, older journals rebuild feature history from their saved raw
events. Any already pending request retains its exact original body for safe retries. Finish such
a pending legacy request against the old model before switching to an 18-feature candidate.


## Delayed feedback and quality metrics

The producer now also computes rolling spending changes and delayed investigation feedback in
an `adaptive_features` block. The default delay is seven simulated days, configurable with
`--feedback-delay-days` on a new journal. See [the evaluation and monitoring guide](ROBUST_EVALUATION.md)
for state migration, delay semantics and the new Grafana quality charts. The live simulator's
private metrics endpoint is enabled automatically in Terraform and the Compose simulation profile.
