MEDIAN
roc_auc: 0.721 ± 0.012 train-validation gap: +0.027 no-model floor: 0.500

MEAN
roc_auc: 0.719 ± 0.013 train-validation gap: +0.029 no-model floor: 0.500

KNN
roc_auc: 0.721 ± 0.013 train-validation gap: +0.030 no-model floor: 0.500

ITERATIVE
roc_auc: 0.721 ± 0.013 train-validation gap: +0.032 no-model floor: 0.500

RANDOM FOREST
roc_auc: 0.720 ± 0.015 train-validation gap: +0.033 no-model floor: 0.500

## Corrected paired comparisons against median

Computed from the saved five-fold validation ROC-AUC scores using `src.evaluate.corrected_paired_test(candidate, median, test_train_ratio=0.25)`. The two-sided test has 4 degrees of freedom. Positive differences favour the candidate. P-values below are unadjusted for the four comparisons.

| Candidate | Mean ROC-AUC difference | Corrected t | Corrected p-value | Folds won |
| --- | ---: | ---: | ---: | ---: |
| mean | -0.001905 | -0.684097 | 0.531490 | 2/5 |
| knn | -0.000172 | -0.063346 | 0.952530 | 2/5 |
| iterative | +0.000016 | 0.008690 | 0.993483 | 2/5 |
| random_forest | -0.000862 | -0.395221 | 0.712851 | 2/5 |

None of the alternatives shows a statistically significant difference from median at the 0.05 level. Retain median: the alternatives add complexity without demonstrating improved ROC-AUC. This does not prove equivalence.

The saved configurations match apart from the imputer, with identical split/fold settings. Target-encoder cross-fitting was unseeded, so these differences also include that source of variation. Training runtimes were not recorded and cannot be recovered from these scores; no models were retrained for this comparison.

Full-precision results, run IDs, and the five fold scores are saved in [validation/w10/imputer_comparison.json](validation/w10/imputer_comparison.json).
