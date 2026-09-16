# Rolling synthetic fraud evaluation

Selected: **v2_hist_gradient_boosting@0.0075**. Rolling FPR constraint met: **True**.

Selection uses rolling development populations only. Final population was generated after freezing the candidate and threshold. No artifact is automatically approved or deployed.

| Candidate / calibration FPR | Mean rolling recall | Worst rolling FPR | Worst FPR 95% upper | Eligible |
| --- | ---: | ---: | ---: | --- |
| v2_logistic@0.005 | 20.28% | 3.75% | 4.03% | False |
| v2_logistic@0.0075 | 21.66% | 5.98% | 6.32% | False |
| v2_logistic@0.01 | 22.29% | 6.91% | 7.27% | False |
| v2_hist_gradient_boosting@0.005 | 19.22% | 0.43% | 0.53% | True |
| v2_hist_gradient_boosting@0.0075 | 20.36% | 0.68% | 0.81% | True |
| v2_hist_gradient_boosting@0.01 | 21.25% | 1.05% | 1.20% | False |
| v2_mlp@0.005 | 19.98% | 2.21% | 2.42% | False |
| v2_mlp@0.0075 | 21.53% | 4.41% | 4.70% | False |
| v2_mlp@0.01 | 22.89% | 5.61% | 5.94% | False |
| v3_logistic@0.005 | 19.86% | 3.12% | 3.38% | False |
| v3_logistic@0.0075 | 21.17% | 4.43% | 4.72% | False |
| v3_logistic@0.01 | 22.13% | 5.41% | 5.74% | False |
| v3_hist_gradient_boosting@0.005 | 19.26% | 1.38% | 1.56% | False |
| v3_hist_gradient_boosting@0.0075 | 20.19% | 1.89% | 2.09% | False |
| v3_hist_gradient_boosting@0.01 | 21.08% | 2.15% | 2.36% | False |
| v3_mlp@0.005 | 20.78% | 5.60% | 5.93% | False |
| v3_mlp@0.0075 | 22.57% | 8.46% | 8.86% | False |
| v3_mlp@0.01 | 23.61% | 11.07% | 11.52% | False |

## Untouched final population

Precision **56.14%**, recall **30.78%**, FPR **0.60%** (6.01 false positives per 1,000 legitimate transactions).
FPR Wilson 95% interval: 0.56%–0.64%. Recall interval: 29.36%–32.24%.

| Fraud scenario | Cases | Detected | Recall |
| --- | ---: | ---: | ---: |
| 1 | 318 | 92 | 28.93% |
| 2 | 2281 | 10 | 0.44% |
| 3 | 1351 | 1114 | 82.46% |

## Interpretation

- v2 has 18 behavioral features; v3 adds rolling spending and delayed confirmation features (28 total). Both MLPs use 32 hidden units, holding capacity fixed.
- Targets 0.5%, 0.75%, 1% apply to calibration only. Selection requires the worst rolling FPR Wilson upper bound to be at most 1%; if none qualifies the least-exceeding candidate is reported without approval.
- Seven-day feedback is revealed only at event time plus seven days. Training labels and calibration labels are subject to the same availability delay. Current hidden labels never enter current features.
- Compromise groups rotate weekly. Seven-day feedback may be stale and can fail to improve terminal detection; the generator was not altered to make this task easier.
- Wilson intervals describe binomial sampling uncertainty, not a guarantee under drift. Repeated customer/terminal observations are correlated; fold variation is also reported and intervals may be optimistic.
- Synthetic populations do not establish real-world performance. Final metrics were not used to revise selection. Do not repeatedly tune against this final population.
- Quality remains measured by scenario: 1 unusual purchases, 2 compromised terminals, 3 customer spending bursts. Scores are not calibrated fraud probabilities.
