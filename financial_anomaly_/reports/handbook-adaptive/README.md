# Delayed-feedback 28-feature candidate

The 28-feature histogram gradient boosting candidate meets the recall and false-negative targets at the measured operating point, but precision remains low. It has not replaced the deployed model.

| Metric | 18 features | 28 features |
|---|---:|---:|
| recall | 82.10% | 86.39% |
| false negative rate | 17.90% | 13.61% |
| precision | 1.02% | 2.80% |
| false positive rate | 71.28% | 26.85% |
| alert rate | 71.38% | 27.38% |

The candidate detects 2,013 of 2,330 fraud cases, misses 317, and flags 70,006 legitimate transactions. Terminal-compromise recall increases from 73.46% to 79.41%. These results compare two newly trained models on identical data; they are not a comparison with the currently deployed artifact.

## Method

The existing ten adaptive features add customer and terminal mean spending over one and seven days, spending ratios, confirmed-fraud rates, and feedback counts. Outcomes enter feature history only seven days after the original transaction. The current transaction outcome is queued after its features are calculated. This is simulated availability of confirmed outcomes for both fraudulent and legitimate transactions; a real deployment would need a corresponding feedback source.

Both models use the same gradient-boosting hyperparameters, chronological partitions and seven-day label-availability gaps. Thresholds maximize validation precision subject to recall strictly above 85%, with no false-positive cap. Artifact exports and validation selection were saved before scoring the test partition. Native and portable predictions were checked for agreement.

Input features are checksum-matched against the prior seven-day-feedback preparation report. Raw data and features were reused without changing labels or feedback delay.

**Evaluation limit:** this September test period was already examined in the prior 18-feature experiment. The comparison measures incremental improvement, but is not a fresh final holdout. Confirm results on multiple rolling windows and an untouched population before promotion. The point estimate reaching 85% is not a guarantee for future transactions.

## Reproduce

Use a new output directory:

```sh
PYTHONPATH=src .venv/bin/python scripts/train_adaptive.py --output-dir reports/handbook-adaptive-new
```

`v3-candidate.json` is the new unapproved 28-feature artifact. `v2-candidate.json` is the matched baseline. `report.json` includes confusion counts, confidence intervals, per-scenario results, hyperparameters, split dates and hashes. `selection-before-test.json` records the frozen validation selection.
