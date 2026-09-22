# Rolling tuning and terminal-risk experiment

This experiment compares the existing 28 features against six additional features:
terminal volume over one and seven days, volume change, confirmed fraud rate over
one day, its difference from the seven-day confirmed fraud rate, and the fraction
of a customer's previous transactions at the current terminal.

Features use prior events only. Fraud feedback becomes available after seven days;
confirmation windows include the exact availability boundary. Initial 14-day
history is excluded from training and evaluation but retained for feature creation.

Six configurations vary iterations (150/250), leaves (7/15/31), minimum leaf size
(50/100), learning rate (0.05/0.08), regularization (1/10), and class weighting
(unweighted, positive weight 10, balanced). Each is evaluated with 28 and 34 inputs
and calibration recall targets of 86%, 90%, and 95%: 36 operating points, each
assessed on three rolling windows.

For evaluation start day E (84, 112, 140), training covers days [14,E-28),
calibration [E-21,E-7), and evaluation [E,E+14). Each gap is seven days to respect
label availability. Model and threshold selection never use fresh-population
outcomes. All rolling evaluation windows must exceed 85% recall; among eligible
operating points selection maximizes mean precision, breaking ties by worst alert
volume. No FPR cap is imposed. If no candidate qualifies, the fallback is explicitly
a diagnostic candidate, not a passing model.

The final fits use Handbook days [14,148), then calibration [155,169). The baseline
is a newly fitted 28-feature balanced model at an 86% calibration recall target,
not the deployed model. Candidate and baseline artifacts are frozen before the two
fresh populations (seeds 2026092101 and 2026092102) are generated and scored. Both
populations use the local generator, 112 days, 500 customers, 50 terminals, seven-day
feedback and 14-day warm-up. They are **cross-generator stress tests**, not untouched
Handbook periods or proof of real-world performance. The existing Handbook test
period was examined previously and must not be advertised as a fresh holdout.

`protocol.json` records settings before fitting; `progress.json` records completed
folds; `selection-before-final.json` freezes rolling selection; and
`artifacts-before-final.json` pins the final artifacts before fresh evaluation.
`report.json` contains all results when the run completes. JSON candidates remain
unapproved. No deployed artifact, producer, API payload or threshold is changed.

The six experimental features are batch research inputs. Portable scoring supports
the 34-feature schema for parity verification; live `FraudScorer` deliberately
continues to reject this schema until a future online feature implementation is
validated. Keep features experimental unless improvements hold across windows and
fresh evaluation.

Reproduce from the project root with an unused output directory:

```sh
PYTHONPATH=src .venv/bin/python scripts/tune_rolling.py \
  --output-dir reports/rolling-tuning-new
```

Input SHA256 must match the recorded seven-day-feedback preparation report.

## Completed results

**Decision: do not promote.** Rolling-window improvements did not generalize to the fresh cross-generator stress tests.

| Rolling comparison | Minimum recall | Mean precision | Mean false-positive rate | Worst alerts / 1,000 |
|---|---:|---:|---:|---:|
| 28-feature baseline | 85.05% | 2.80% | 28.21% | 375.1 |
| Best tuned 28 features | 85.13% | 2.86% | 27.91% | 377.6 |
| Selected 34 features | 85.05% | 3.28% | 23.64% | 281.8 |

Selected configuration: 250 iterations, 7 leaves, minimum leaf size 100, learning rate 0.05, L2 regularization 10, no class weighting, and calibration recall target 86%. Added features improve precision in two of three windows relative to the best tuned 28-feature candidate, not consistently in all windows. They remain experimental.

| Fresh population | Model | Recall | False-negative rate | Precision | False-positive rate |
|---|---|---:|---:|---:|---:|
| 2026092101 (135,734 rows) | baseline | 60.47% | 39.53% | 3.37% | 44.85% |
| 2026092101 (135,734 rows) | candidate | 70.23% | 29.77% | 3.32% | 52.96% |
| 2026092102 (139,248 rows) | baseline | 55.13% | 44.87% | 3.17% | 44.56% |
| 2026092102 (139,248 rows) | candidate | 63.18% | 36.82% | 3.13% | 51.74% |

Fresh-population recall fails the 85% target in both populations, and precision is slightly below the matched baseline. These populations use the local generator rather than the Handbook generator, so the result shows transfer limitations and cannot be presented as same-distribution Handbook performance.

The next evaluation should use fresh populations from the original Handbook generator, or representative target application data, before any release decision. Further tuning against these two now-observed populations would require another untouched final evaluation.

Validation: 15 targeted tests passed, covering time-window boundaries, unavailable/future outcomes, terminal isolation, chronological gaps, strict recall eligibility, timestamp resolution, portable/native parity, active-model compatibility, and rejection of the experimental schema by live scoring. Lint and diff checks pass. The deployed artifact remains unchanged.
