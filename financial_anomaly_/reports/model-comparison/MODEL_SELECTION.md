# Synthetic fraud model selection

Selected candidate: **v2_mlp**.

Selection maximizes validation recall under the predeclared false-positive cap. Ties use precision, then average precision. Test and external labels never select a model or threshold.

| Candidate | Features | Validation recall | Test precision | Test recall | Test FPR | Test AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v1_logistic | 4 | 0.336 | 0.244 | 0.348 | 1.391% | 0.219 |
| v1_hist_gradient_boosting | 4 | 0.357 | 0.290 | 0.348 | 1.098% | 0.309 |
| v1_mlp | 4 | 0.363 | 0.227 | 0.368 | 1.621% | 0.256 |
| v2_logistic | 18 | 0.493 | 0.362 | 0.538 | 1.228% | 0.500 |
| v2_hist_gradient_boosting | 18 | 0.475 | 0.285 | 0.494 | 1.605% | 0.488 |
| v2_mlp | 18 | 0.517 | 0.340 | 0.568 | 1.427% | 0.549 |

Selected model meets the 1.0% FPR target on test: **False**. This is an observed result, not a guarantee on new traffic.

## Selected model by fraud scenario

| Scenario | Fraud cases | Detected | Missed | Recall |
| --- | ---: | ---: | ---: | ---: |
| 1 | 79 | 23 | 56 | 0.291 |
| 2 | 130 | 2 | 128 | 0.015 |
| 3 | 291 | 259 | 32 | 0.890 |

## Independent-population evaluation

These additional data are evaluated only after selection, with frozen models and thresholds. No refitting or threshold adjustment occurs.

| Candidate | Precision | Recall | FPR | AP |
| --- | ---: | ---: | ---: | ---: |
| v1_logistic | 0.452 | 0.173 | 0.583% | 0.177 |
| v1_hist_gradient_boosting | 0.408 | 0.216 | 0.873% | 0.234 |
| v1_mlp | 0.428 | 0.198 | 0.737% | 0.206 |
| v2_logistic | 0.568 | 0.278 | 0.588% | 0.315 |
| v2_hist_gradient_boosting | 0.549 | 0.292 | 0.668% | 0.331 |
| v2_mlp | 0.566 | 0.309 | 0.659% | 0.342 |

## Interpretation and limitations

- Scenario 1: unusual purchases; scenario 2: compromised terminals; scenario 3: customer spending bursts.
- Features use only prior unlabeled observations. Labels, scenario codes and IDs are excluded from model inputs; scaling is fit on training only.
- Terminal fraud can look identical to legitimate purchases in this generator. Label-free terminal activity does not reveal which terminal was secretly compromised. Perfect recall is not a realistic goal.
- The neural networks use 8 hidden units (v1) or 32 (v2); tree and logistic settings are fixed across feature sets. The MLP feature ablation also changes capacity and is not a pure feature-only experiment.
- Existing test-period data were previously used for an earlier candidate report. The additional seeded population provides a fresh check; neither synthetic set establishes real-world performance.
- Class weighting changes score calibration. Scores are ranking signals, not demonstrated real-world fraud probabilities.
- selected-model.json is a data-only portable artifact, with native/portable parity checked. It is not production-approved or automatically deployed. Existing production release gates are unchanged.
- comparison.json records split boundaries, confusion counts, scenario recall, daily precision@100, runtime, hyperparameters, dependency versions and input/policy hashes. selection-before-test.json freezes the choice before test scoring.
