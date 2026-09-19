# Active demo model benchmark

The demo now uses `robust-v2_hist_gradient_boosting-dadcc90775b3`, an 18-feature histogram
gradient-boosted classifier. The model bytes match the previously evaluated candidate exactly;
threshold 0.8280339469486367. The four-feature MLP artifact was removed. This is a user-authorized
demo promotion, not production approval.

Fresh synthetic population: seed 1909, 500 customers, 50 terminals, 112 days, 161,833 transactions.
No retraining or threshold tuning occurred.

| Metric | Result |
| --- | ---: |
| False-positive rate | 0.7357% |
| Precision | 48.8722% |
| Recall | 23.5407% |
| Accuracy | 97.0680% |
| False positives per 1,000 legitimate transactions | 7.3565 |
| Average precision | 0.2672 |

False positives remain below 1%, but precision is below the provisional 50% review-alert target
and recall remains weak. Scenario recall: unusual purchases 26.54%, compromised terminals 0.19%,
spending bursts 81.30%. The FPR Wilson 95% interval is 0.6946%–0.7791%; repeated customer/terminal
observations can make binomial intervals optimistic. These are synthetic results, not real-world
acceptance evidence. Prior reports used different populations and are not directly comparable.

Local in-process scoring p95 was 0.091 ms, excluding network/persistence. A separate localhost HTTP
benchmark against the actual Terraform deployment completed 500/500 new requests successfully,
all with the expected model version. Sequential throughput was 189.1 requests/second, p50 5.06 ms,
p95 5.88 ms, p99 6.66 ms, including ledger persistence. This is not a maximum-load or HA benchmark.
Benchmark requests are saved in the ledger under unique `benchmark-*` IDs and are separate from
the simulator's ground-truth metrics. The simulator continues running against the new model.

Machine-readable evidence: `benchmark.json` (quality/local scoring), `api.json` (deployed API).

```bash
.venv/bin/python scripts/benchmark_model.py --seed 1909 --output-dir reports/benchmark-repeat
.venv/bin/python scripts/benchmark_api.py --count 500 --output reports/benchmark-repeat/api.json
```

Use new output paths. Repeating a seed reproduces a population; use a new, reserved seed for a
future independent evaluation. Scripts refuse to overwrite existing reports.
