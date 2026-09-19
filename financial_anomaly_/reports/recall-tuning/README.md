# Recall tuning experiment

**Result: the requested recall above 85%, false-negative rate below 15%, and improved precision were not achieved on the fresh test population. The active model was not replaced.**

Eight gradient-boosting configurations were evaluated at three calibration recall targets (86%, 90%, 95%). No false-positive cap was imposed. Selection maximized validation precision among candidates with recall strictly above 85%. Parameters and the selected threshold were saved before final testing.

The final benchmark contains 160,084 synthetic transactions. All rows below use the same final population. Calibration FPR targets are not guarantees on new data.

| Operating point | Recall | False-negative rate | Precision | False-positive rate | Transactions alerted |
|---|---:|---:|---:|---:|---:|
| Current model, current threshold | 33.94% | 66.06% | 49.60% | 0.68% | 1.31% |
| Current model, 1% calibration FPR target | 35.34% | 64.66% | 39.59% | 1.06% | 1.71% |
| Current model, 5% calibration FPR target | 39.11% | 60.89% | 15.29% | 4.24% | 4.91% |
| Current model, 10% calibration FPR target | 48.71% | 51.29% | 6.33% | 14.12% | 14.78% |
| Selected tuned candidate | 53.89% | 46.11% | 4.25% | 23.76% | 24.34% |

The selected candidate achieved 92.65% recall on selection validation, but only 53.89% on the independent final population. Validation covered the last seven days of one population; final evaluation covers all 112 days of another population, including initial history accumulation. This difference and population variation limit generalization. Selection validation is not an unbiased performance estimate.

The candidate generated 37,306 false alerts and missed 1,417 fraudulent transactions. Its terminal-compromise recall was 21.10%, versus 79.94% for unusual purchases and 92.33% for customer bursts. In this simulator, terminal compromise can change labels without changing transaction amount or time; hidden compromise membership also changes weekly. These results do not establish that further tuning can meet the combined targets using the existing 18 features.

Recommended next experiment: evaluate across several independent populations and rolling time windows with consistent warm-up periods; then test additional observable risk signals or realistically faster confirmed-fraud feedback. Keep final evaluation separate from selection. Do not change synthetic labels or expose future outcomes to inflate recall.

Artifacts: `selection-before-test.json` records development selection; `candidate.json` is an unapproved experiment; `report.json` contains metrics, confusion counts, intervals, scenarios, thresholds, and hashes. The active model hash was verified unchanged after recovery.

Reproduce from the project root with an unused output directory:

```sh
PYTHONPATH=src .venv/bin/python scripts/tune_recall.py --output-dir reports/recall-tuning-new
```

Population seeds: training 3141, development 3142, final 3143. Training uses days [0,77), calibration [84,98), and selection validation [105,112). Final starts 2025-07-01. Dataset policy comes from `configs/robust-evaluation.json`. The script never promotes or deploys its candidate.
