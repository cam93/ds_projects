# Official Handbook reproduction and aligned population evaluation

This experiment separates three questions:

1. Can the local feature pipeline reproduce the published benchmark?
2. Which model performs best under the application's high-recall requirement?
3. Does the frozen model transfer to a new population from the original generator?

The reference source revision is
`81cf7d1714bb7b2f5b496407d9055d91dc68dc25`; raw files use
`6e67dbd0a3bfe0d7ec33abc4bce5f37cd4ff0d6a`. Source functions were isolated from
notebook top-level execution and retained in `scripts/handbook_reference`, together
with the upstream GPL-3.0 license. Notebook feature and generator downloads were
verified against the pinned revision.

## Feature and evaluation conventions

The batch reference schema contains 15 inputs: amount, weekend/night flags,
customer transaction counts/mean amounts over 1/7/30 days, and terminal delayed
transaction counts/fraud rates over 1/7/30 days. Customer windows include the
current observed transaction; the lower boundary is excluded. Terminal windows
are `(now - 7 days - window, now - 7 days]`. The current label is excluded.

The reference comparison retains zero-amount records and uses the exact upstream
known-compromised-card removal function. Training: July 25–31, 2018. Evaluation:
August 8–14, 2018. The report separately scores all transactions for those dates.
Reference classification metrics use threshold 0.5; AP, ROC AUC and Card
Precision@100 are ranking metrics, not precision at an 85% recall threshold.
The card metric includes the reference's removal of previously detected cards.

Application comparison retains all positive-amount transactions with no known-card
exclusion. Two chronological windows have 28-day training, seven-day gaps,
14-day calibration and 14-day evaluation. Six model configurations cover histogram
gradient boosting, Random Forest and XGBoost, including regularized alternatives.
Thresholds maximize calibration precision at recall >=86%. Selection requires
recall >85% in both evaluation windows and maximizes mean precision. A fallback is
explicitly diagnostic if no configuration meets the requirement.

Final fits and thresholds are frozen before fresh population generation. The
fresh population uses the original Handbook generation and fraud mechanisms with
5,000 customers, 10,000 terminals, radius 5, 112 days and new profile seeds. Customer
IDs are offset to give the upstream per-customer RNG fresh seeds. The fraud
mechanism retains upstream deterministic day-based sampling. Initial 37 days are
warm-up. These are synthetic population-transfer results, not real-world results.

## Run

Install the existing training dependencies and the optional benchmark extra:

```sh
.venv/bin/pip install -e '.[training,benchmark]'
PYTHONPATH=src:. .venv/bin/python scripts/benchmark_handbook.py \
  --output-dir reports/handbook-reference-new
```

XGBoost needs OpenMP on macOS. This run used the existing scikit-learn bundled
runtime by setting `DYLD_LIBRARY_PATH` to the virtual environment's
`lib/python3.10/site-packages/sklearn/.dylibs` directory. No system license was
accepted or changed. The XGBoost version is pinned in the benchmark extra; all
library versions are recorded in the report. Version differences may affect
reproduction of the historical XGBoost defaults.

`reference.json` contains published versus measured baseline results.
`selection-before-final.json` freezes model selection;
`artifacts-before-final.json` pins the final offline artifacts and thresholds;
`report.json` contains fresh-population evaluation. Native joblib artifacts are
stored under the ignored `models/artifacts` directory, never loaded by the live API.
Only load these artifacts from trusted local runs. No deployed model changes.

References:
- https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/BaselineFeatureTransformation.html
- https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/BaselineModeling.html

## Completed results

The reference feature check agrees with upstream calculations (maximum absolute customer-feature error 5.23e-12; terminal-feature error zero). The depth-two decision tree and Random Forest reproduce the published ranking results to the published precision. This verifies the reference pipeline; it does not imply that high-recall precision equals average precision.

| Reference model | Measured AP | Published AP | Measured Card Precision@100 | Published CP@100 |
|---|---:|---:|---:|---:|
| decision_tree_2 | 0.496 | 0.496 | 0.241 | 0.241 |
| random_forest | 0.658 | 0.658 | 0.287 | 0.287 |
| xgboost | 0.563 | 0.639 | 0.246 | 0.273 |

Historical XGBoost defaults/results were not exactly reproduced with XGBoost 3.0.5. This version difference is reported rather than tuned against the reference test to force a match.

Regularized XGBoost won the all-transaction rolling comparison. It uses 250 trees, depth 3, learning rate 0.05, minimum child weight 5 and L2 regularization 10, with a validation-selected threshold. A default Random Forest needed threshold zero to reach the calibration target and therefore flagged every transaction: strong ranking performance alone does not ensure a useful high-recall operating point.

Fresh evaluation contains **723,227 transactions** after the 37-day warm-up. Both models are newly trained on the official 15 inputs; neither row represents the currently deployed artifact.

| Fresh model | Recall | False-negative rate | Precision | False-positive rate | Average precision | Alerts / 1,000 |
|---|---:|---:|---:|---:|---:|---:|
| xgb_regularized | 85.29% | 14.71% | 3.39% | 21.74% | 73.56% | 223.0 |
| hist_gradient_boosting | 85.30% | 14.70% | 3.33% | 22.15% | 70.61% | 227.1 |

**Assessment:** the selected model meets the recall/FNR targets as point estimates on one fresh aligned synthetic population. Precision remains low, so it is not approved for deployment. Relative to histogram gradient boosting on the same full features, XGBoost provides only a modest high-recall improvement. The earlier custom-generator stress tests measured a different distribution; their lower recall was not a valid estimate of fresh Handbook performance.

The data supports useful ranking, but this experiment does not prove an absolute performance ceiling. The remaining decision is an explicit precision/alert-volume budget. Evaluate calibration-selected operating points against that budget and investigate missed fraud by scenario; repeat on additional untouched populations before promotion. Real-time use also requires the official feature conventions to be implemented and parity-tested in the producer; these changes are batch evaluation only.

Checks: seven targeted feature, data preparation and active-model tests passed; lint and diff checks passed; offline artifact hashes verified; deployed model bytes unchanged.
