# Stable false-positive evaluation and live quality monitoring

The new workflow measures model stability across three synthetic populations and two chronological
windows per population. It compares 18-feature (v2) and 28-feature (v3) logistic regression,
histogram gradient boosting and neural networks, each at 0.5%, 0.75% and 1% calibration FPR targets.
Both neural networks have 32 hidden units, making their feature comparison use the same capacity.

The completed run selected **v2 histogram gradient boosting at the 0.75% calibration target**.
Its worst rolling FPR was 0.68%, with a Wilson 95% upper bound of 0.81%. On the untouched final
population it achieved **0.60% FPR, 56.14% precision and 30.78% recall**. The FPR interval was
0.56%–0.64%. These are synthetic results on different data from the earlier report; they are not
an apples-to-apples improvement estimate. See [the full report](../reports/robust-evaluation/REPORT.md).

The ten additional features were implemented and evaluated, but did not win the stability test.
Terminal compromise remains difficult: 0.44% final recall versus 82.46% for spending bursts.
Weekly compromises rotate before seven-day investigation feedback arrives. We have retained that
limitation rather than changing fraud labels or the generator to inflate detection scores.

## Run the comparison

```bash
.venv/bin/python -m pip install -c constraints.txt -e '.[training]'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/robust_evaluation.py \
  --policy configs/robust-evaluation.json --output-dir reports/robust-evaluation-new
```

Use a new output directory. The script refuses existing directories. It generates 112 simulated
days per population with 500 customers and 50 terminals. Development seeds are 41, 53 and 67;
final seed is 907. Configuration, feature hashes and dependency versions are saved. Repeating this
exact run checks reproducibility; it does not produce a new independent evaluation. Future model
development must reserve another unseen seed before examining results.

For each development seed, training ends at day 42 or 63, calibration spans the following 14 days,
and evaluation begins after the seven-day label-availability delay. Training also excludes labels
that would not yet have been investigated when fitting starts. Features always use prior events.
The new populations are separate; their customer histories are never mixed when preparing features.

Candidates qualify only if the **upper Wilson 95% FPR bound in every rolling evaluation window**
is at most 1%. Among qualifying candidates, mean recall wins, followed by precision. If none
qualifies, the least-exceeding candidate is reported explicitly as failing the rolling constraint.
Wilson intervals assume binomial observations; customer and terminal correlation can make them
optimistic. The per-population/window results expose variation that an aggregate interval hides.

The selected family is refit on pooled development data before day 77 (day 84 minus the delay).
Its threshold is calibrated on days 84–98, whose labels are available by day 105. The model,
threshold and selection report are written **before the final population is generated**. Final
inference uses the actual portable artifact and is checked against native predictions. No final
result changes the selected model, threshold or production approval.

Outputs include `evaluation.json`, `selection-before-final.json`, `REPORT.md`, a portable
`selected-model.json` and its SHA256, and local `final-predictions.csv`. The completed run's large
prediction file is excluded from Git. Production approval remains false and the existing deployed
model is retained. Evaluation does not deploy or change Terraform's model path.

## Features and delayed feedback

`features/adaptive.py` adds customer and terminal mean amounts over preceding one-day and seven-day
windows, ratios of those means, and customer/terminal confirmed fraud rates and feedback counts.
These ten features supplement the existing 18. Rolling windows include their lower boundary and
exclude the current transaction. An empty spending history has mean zero and change ratio one.

The simulator schedules each known synthetic outcome to become available after **seven simulated
days**. Feature feedback rates cover confirmations received in the preceding seven days and include
both genuine and fraudulent outcomes. Before any confirmation, rate/count are zero. The current
outcome is queued only after its features are computed. No current hidden fraud label, scenario,
customer ID or terminal ID becomes a numeric model input.

The new `adaptive_features` API block includes the feedback delay. A v3 model requires both feature
blocks and a delay matching its trained configuration; missing/mismatched fields return 422.
Legacy v1/v2 requests and idempotency hashes remain compatible. Producers remain authenticated
and trusted. A real system would need an investigation-result source, not synthetic oracle labels.

The delay is configurable in the evaluation policy, preparation and standalone streaming:

```bash
.venv/bin/python scripts/prepare_dataset.py data/raw/transactions.csv \
  --output-dir data/processed/transactions-v3 --feedback-delay-days 7
.venv/bin/python scripts/simulate.py stream --feedback-delay-days 7
```

An existing journal must keep its original delay; use a new state file for a different experiment.
Older journals rebuild the additional state from their saved events once. Pending requests retain
their exact original bodies, so finish older pending requests before switching to a model requiring
new fields. Rolling state and delayed outcomes are checkpointed with the pending request before
HTTP submission. This is single-producer demo state, not a distributed production feature store.

## Quality dashboard

After the usual `terraform apply` (or Compose with the `simulation` profile), the dashboard includes:

- Precision and recall by model version.
- False positives per 1,000 legitimate transactions.
- Recall by fraud scenario and model version.
- The number of acknowledged transactions evaluated.

These charts use **immediate synthetic oracle labels for evaluation only**. Model input feedback
still waits seven simulated days. Results are cumulative over the retained simulator journal,
not a rolling quality estimate restricted to Grafana's selected time range. A zero denominator
has no defined value and appears as a gap. Model versions are kept separate across model changes.

Quality counts are updated atomically with acknowledgement, so retries do not double-count.
Existing journals rebuild the aggregates once; restarting containers preserves them. Destroying
the state volume starts a new evaluation history. Raw labels, customer IDs and transaction IDs are
not published in metrics. Metrics come only from acknowledged simulator events, not file replay.

The simulator serves `/metrics` on private port 8002 with the existing metrics bearer key.
Prometheus scrapes it as `fraud-demo-quality`; no host port is published. Anonymous Grafana remains
Viewer-only. Standalone metrics are disabled unless `--metrics-port` is specified; when enabling
them, supply `--metrics-key-file` and restrict network access. If simulation is disabled, its scrape
target is down and current quality charts have no samples; the API remains independently monitored.
Prometheus can retain historical samples after the simulator stops.

## Verify without deploying the candidate

```bash
docker build -f deploy/docker/api.Dockerfile -t fraud-detector-review:local .
.venv/bin/python scripts/smoke_compose.py --model reports/robust-evaluation/selected-model.json
```

The isolated smoke stack checks scoring, replay, simulator restart, authenticated metrics,
Prometheus quality samples and anonymous read-only Grafana, then removes its resources. Unit tests
cover delayed feedback boundaries, online/offline parity, legacy migration, duplicate acknowledgements,
threshold selection, label-availability cutoffs, and feature/feedback-delay validation.
