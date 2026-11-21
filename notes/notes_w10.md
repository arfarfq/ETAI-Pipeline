# Week 10: missing values and deploying a fitted pipeline

These notes explain [notebook 10: Missing values](../W10_Notebooks/10_missing_values.ipynb) and [notebook 11: Deployment](../W10_Notebooks/11_deployment.ipynb). Numerical results in the explanatory sections are the notebooks’ saved example outputs, not experiments rerun on this project. Section 11 records actual project progress and results checked on 20 November 2026.

## 1. The central problem

A predictive system must handle the information available **when a prediction is requested**. A new person may have incomplete records, an unfamiliar category, and no known outcome. The preprocessing must still produce inputs the trained model can use.

This links the two notebooks:

1. Choose an imputation procedure that can learn from training features and fill gaps in unseen features.
2. Evaluate the entire procedure without letting validation information influence training.
3. Save the fitted preprocessing and classifier together.
4. Apply the saved procedure to new rows without training again.

An **imputer** estimates missing feature values. It does not recover them with certainty. The practical aim is to support useful, honestly evaluated predictions while preserving information about the original gaps.

## 2. Understanding missingness before choosing a method

### 2.1 Missing values must first be recognised

A gap may already be represented by `NaN`, or disguised as `"?"`, an empty string, or an invalid value. Cleaning should standardise placeholders, convert numeric text, apply configured domain rules, and canonicalise category spellings. For example, `priors_count = "?"` becomes a numeric missing value.

Missingness indicators must be computed **after this cleaning and before imputation**. Otherwise a disguised gap can be missed, or filling the gap can erase the evidence that it existed.

### 2.2 MCAR, MAR, and MNAR

Let `R_j = 1` mean feature `j` is missing. The missingness mechanism describes how that event relates to the data.

| Mechanism                          | Meaning                                                                                        | Example                                                                                                             |
| ---------------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| MCAR: missing completely at random | Missingness is independent of both observed and unobserved values.                             | A random file-transfer failure removes some ages.                                                                   |
| MAR: missing at random             | Conditional on observed information, missingness does not further depend on the missing value. | Age is more often absent in one observed recording system; within each system, absence does not depend on age.      |
| MNAR: missing not at random        | Missingness still depends on the missing value after accounting for observed information.      | Higher prior-offence counts are systematically less likely to be recorded, even among otherwise comparable records. |

MAR does **not** mean that gaps are scattered randomly throughout the table. It means the mechanism can be explained by the observed information under the assumed model.

Comparing missingness with observed features or outcomes can reveal associations inconsistent with MCAR. It generally cannot establish MNAR rather than MAR: the missing values themselves are unavailable. The configuration name `mnar_indicator_sources` is a project convention, not proof of a missingness mechanism.

### 2.3 What the notebook finds

The example development set contains 5,771 rows. Its missingness summary includes:

| Feature           | Missing entries | Missing percentage | Positive-outcome rate if missing | Rate if present |
| ----------------- | --------------: | -----------------: | -------------------------------: | --------------: |
| `priors_count`    |             397 |               6.9% |                            0.350 |           0.458 |
| `c_charge_degree` |             192 |               3.3% |                            0.594 |           0.446 |
| `juv_fel_count`   |             174 |               3.0% |                            0.448 |           0.451 |
| `age`             |             120 |               2.1% |                            0.492 |           0.450 |
| `sex`             |              84 |               1.5% |                            0.452 |           0.451 |

Missing `priors_count` and `c_charge_degree` have particularly noticeable outcome associations. This motivates retaining their missingness flags. These are descriptive associations, not causal effects or statistical proof; smaller observed differences also need uncertainty considered.

### 2.4 Why deleting rows or columns is unattractive

**Complete-case deletion** retains only rows with every selected feature present. Here it retains 4,858 of 5,771 rows, approximately 84%, losing about one person in six. The outcome rate barely changes: 0.451 overall versus 0.453 in complete rows.

A similar overall outcome rate does not establish that deletion is unbiased. Feature relationships, group representation, or performance for incomplete records could still change. Deletion also reduces the training sample. At deployment, dropping incomplete rows violates this exercise’s requirement that every input row receive a prediction.

Dropping a column sacrifices its observed information too. This is especially unattractive for `priors_count`, which the notebook identifies as an important predictor and which is present for most people.

### 2.5 Preserve the fact that a value was missing

For an original feature `x_j`, create:

```text
x_j_was_missing = 1 if the cleaned value is missing, otherwise 0
```

Then impute `x_j` and retain both inputs. A recorded count of 2 and a missing count filled with 2 now remain distinguishable.

The indicator uses information available at prediction time and needs no outcome label. It can preserve predictive signal, but it does not solve MNAR bias or guarantee fairness. Its usefulness also depends on the recording process remaining sufficiently similar after deployment.

## 3. Leakage and the label trap

### 3.1 What honest cross-validation requires

Cross-validation repeatedly separates development data into training and validation folds. For each split, fit every learned preprocessing step and the classifier on the training fold, then transform and score the validation fold using those fitted objects.

```text
Training fold:   fit scaler → fit imputer → fit encoder/selector → fit classifier
Validation fold: apply those fitted steps → predict → compare with validation labels
```

In practice the numeric and categorical transformations run in parallel branches. A `Pipeline` containing a `ColumnTransformer` keeps their fitting inside each fold.

Fitting even a feature-only imputer on all development rows before cross-validation leaks validation-feature information into training. Using validation labels is more severe: it feeds the answers into the input preparation.

Fixed, predefined cleaning rules can be applied consistently without learning from the validation distribution. Rules or thresholds estimated from the data need the same fold discipline as other learned transformations.

### 3.2 Why filling by class produces misleading scores

Class-conditional imputation fills a gap using the average or mode among people with the same outcome label. In the notebook, a missing `priors_count` becomes:

| Actual label | Filled value |
| ------------ | -----------: |
| 0            |         2.19 |
| 1            |         4.87 |

When this is done before cross-validation, each validation row’s label determines its feature value. The classifier can learn to recognise these values as clues to the answer.

The notebook’s ROC-AUC rises from **0.724** with honest median imputation to **0.790** with this leaking procedure. This is an invalid comparison, not evidence of better generalisation.

Learning class means inside each training fold would remove the use of validation labels when estimating those means. It still would not tell us which mean to use for a validation or new person whose outcome is unknown. A deployable prediction procedure must resolve that issue without consulting the true label.

### 3.3 Supervised imputation and forest proximities

A forest trained to predict the outcome can define similarity through its leaves: two people are close if they often land in the same leaf. Because the forest learned from labels, this representation must be trained within each training fold.

Some class-dependent forest methods handle an unseen person by considering one copy for each possible class. That requires a complete procedure for evaluating and combining those possibilities; copying the row alone does not solve the problem.

This week’s standard imputers avoid the issue by learning the missing features from **other features**, without the recidivism label. This restriction concerns the imputer: the classifier still learns from labels, and a properly cross-fitted target encoder can also use training labels while transforming unseen rows without their labels.

## 4. The imputation methods

### 4.1 Mean, median, and mode: `SimpleImputer`

These methods estimate one replacement per column from its observed training values:

- **Mean:** the arithmetic average for a numeric column.
- **Median:** the middle numeric value, less affected by extreme observations.
- **Mode:** the most frequent category.

They are fast, easy to reproduce, and require no relationships between columns. They also fill every gap in a column with the same value, compressing variation and potentially distorting associations.

The mean minimises squared error for a constant prediction; the median minimises absolute error. Consequently, median imputation is a particularly relevant baseline when evaluating reconstruction using mean absolute error.

### 4.2 K-nearest neighbours: `KNNImputer`

KNN estimates a missing entry using nearby training rows that have an observed value for that feature. The notebook uses the default uniform weighting, so available donor values are averaged.

Distances are computed using features observed in both rows, with an adjustment for how many features overlap. Rows with little overlap provide weaker evidence of similarity. If useful neighbours are unavailable, the implementation can fall back to the training-column mean.

The main setting is `n_neighbors`:

- A small value is more local but sensitive to individual donors.
- A large value smooths more strongly and may combine dissimilar people.

Distance is sensitive to units: a column with a much larger numerical range can dominate. Scaling makes comparisons more meaningful. KNN can also be expensive because transformation requires comparisons with training observations. This notebook uses it for numeric columns; it does not provide categorical mode imputation.

### 4.3 Iterative imputation: chained conditional models

`IterativeImputer` fills each incomplete numeric feature by predicting it from the other numeric features:

1. Initialise gaps, by default with column means.
2. Select an incomplete feature as a temporary regression target.
3. Fit a regressor on rows where that feature is observed, using the other features as inputs.
4. Predict its missing entries.
5. Repeat for other incomplete features and cycle through the process.

Here the temporary target is a **feature**, such as age, not the classification outcome. Updated estimates allow information to flow between incomplete columns.

The default regressor is Bayesian ridge regression. With `sample_posterior=False`, the method produces one completed dataset using point predictions. `max_iter` limits the number of rounds; the default deterministic procedure can also stop early according to its convergence tolerance.

Its success depends on informative relationships and suitable regression assumptions. Predictions for counts may be fractional or outside plausible bounds; an estimated count is not necessarily a recovered integer measurement.

### 4.4 Forest-based iterative imputation

Replacing the regressor with `RandomForestRegressor` uses the same chained loop but learns nonlinear relationships and interactions:

```python
IterativeImputer(
    estimator=RandomForestRegressor(
        n_estimators=50,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    ),
    max_iter=10,
    random_state=42,
)
```

A separate regression forest is fitted for each imputation target during the rounds. The cost therefore grows with the number of columns, rounds, and trees.

The notebook calls this the **missForest idea**. It is a numeric forest-based implementation using scikit-learn’s iterative framework, rather than an exact reproduction of every aspect of the original mixed-type missForest algorithm. It also differs from outcome-trained forest proximities: it predicts each missing feature directly from the others.

### 4.5 Method comparison

| Method                         | Uses other features? | Main strength                                      | Main limitation                                             |
| ------------------------------ | -------------------- | -------------------------------------------------- | ----------------------------------------------------------- |
| Mean/median/mode               | No                   | Cheap, stable baseline                             | Ignores relationships and uncertainty                       |
| KNN                            | Yes                  | Local similarity without a global regression model | Needs meaningful distances; potentially expensive           |
| Default iterative              | Yes                  | Combines information across columns                | Depends on regression assumptions and available predictors  |
| Forest iterative               | Yes                  | Learns nonlinear relationships                     | High computational cost; still needs informative predictors |
| Multiple posterior imputations | Yes                  | Represents variation across plausible fills        | Multiple pipelines and more computation                     |

## 5. How these methods fit into preprocessing

Notebook 10 introduces a factory registry, `_NUMERIC_IMPUTERS`, selected through `preprocessing.imputation.numeric_strategy`. The available settings are `median`, `mean`, `knn`, `iterative`, and `random_forest`.

The resulting preprocessing branches are:

```text
Numeric columns      → scale → impute ─────────────┐
Categorical columns  → impute → encode ────────────┼→ combine → select → classifier
Missingness flags    → pass through ──────────────┘
```

The numeric order changes to **scale first, then impute**. The configured standard, min–max, and robust scalers can learn from observed entries while retaining gaps for the imputer. Each is fitted within the training fold. Robust scaling centres on the median and scales using the interquartile range, reducing the influence of extreme counts on the scale.

This ordering supplies scaled inputs to KNN. Scaling is less central for a forest, but a consistent branch lets the imputer be changed through configuration. Choosing `scaler: none` removes the distance-scaling benefit for KNN.

The branches also restrict information: the numeric imputer only sees the columns sent to its branch. A categorical feature included elsewhere in the classifier does **not** automatically help numeric imputation.

## 6. Evaluating whether the guesses are good

### 6.1 Hide values whose truth is known

Real gaps have no observed answer. The notebook instead takes the first CV split, hides 200 known validation values in each of `age` and `priors_count`, and evaluates reconstruction:

1. Fit scaling and imputation on training rows only.
2. Transform validation rows containing the artificial gaps.
3. Invert the scaling to recover original units.
4. Score only the deliberately hidden entries against their saved truth.

The metric is mean absolute error:

```text
MAE = sum of |imputed value − true hidden value| / number of hidden values
```

An age MAE of 9 means an average absolute error of nine years, not nine percent.

### 6.2 Results when only numeric features are available

| Hidden feature | Median | Mean |   KNN | Iterative | Forest iterative |
| -------------- | -----: | ---: | ----: | --------: | ---------------: |
| `age`          |   9.16 | 9.26 | 10.87 |      8.96 |             8.97 |
| `priors_count` |   3.08 | 3.47 |  3.74 |      3.26 |             3.06 |

The elaborate methods offer little improvement over the median. The five numeric features contain limited information about one another: juvenile counts, for example, do not determine someone’s current age.

### 6.3 Give the imputers useful information

The notebook adds a numeric representation of `age_cat`: 0 for under 25, 1 for 25–45, and 2 for over 45. This preserves ordering, although it does not represent actual ages or equal-width intervals.

| Hidden feature | Median | Mean |  KNN | Iterative | Forest iterative |
| -------------- | -----: | ---: | ---: | --------: | ---------------: |
| `age`          |   9.16 | 9.26 | 4.34 |      4.50 |             4.03 |
| `priors_count` |   3.08 | 3.47 | 3.98 |      3.16 |             3.10 |

Age errors approximately halve for the multivariate methods. The univariate methods cannot use `age_cat`, so their age errors remain unchanged. Prior-count estimation remains difficult.

The lesson is that **the information supplied to an imputer can matter more than its algorithm**. This experiment supplies `age_cat` to the reconstruction benchmark; it does not itself redesign the full pipeline’s separate branches.

### 6.4 Reconstruction and prediction answer different questions

Reconstruction MAE asks how accurately missing features are estimated. Classification ROC-AUC asks how well the complete system ranks positive outcomes above negative outcomes. A better reconstructed age need not improve ROC-AUC, especially if the classifier already receives `age_cat`.

Randomly hiding observed values also does not necessarily reproduce the mechanism of real gaps. The experiment is a controlled diagnostic, not proof of performance on naturally missing values. Its one split and one masking seed limit how much to infer from small differences.

## 7. Multiple imputation and uncertainty

### 7.1 Why a single fill hides uncertainty

Replacing a missing count with 3 supplies one plausible value while concealing that 1 or 8 might also be plausible. Multiple imputation produces `m` completed datasets with different draws for missing entries, fits a model to each, and combines the results.

In the notebook, different seeds and posterior sampling provide those draws:

```python
IterativeImputer(sample_posterior=True, random_state=seed)
```

The default Bayesian ridge estimator supports the predictive uncertainty needed for this sampling. An arbitrary replacement estimator may not support it; the forest regressor above cannot simply be given the same sampling option.

### 7.2 Combining predictions

For a new person `x`, the ensemble probability is:

```text
p_average(x) = (p_1(x) + ... + p_m(x)) / m
prediction = 1 if p_average(x) >= 0.5, otherwise 0
```

Average **probabilities before thresholding**, retaining the models’ confidence information.

For one missing prior count, the notebook displays draws of `[2.2, 8.8, 1.4, 4.1, 3.9]`. Their spread illustrates imputation uncertainty under the assumed models. It is not a calibrated confidence interval for that person’s true count, and it does not include every source of uncertainty.

The example’s five individual out-of-fold ROC-AUC values are approximately 0.720–0.724; their averaged probabilities achieve 0.725. This small improvement is illustrative. Averaging does **not** guarantee an improvement over each individual model or the median baseline.

Training and storage grow roughly with `m`, as does inference when all pipelines are retained. Posterior sampling also introduces randomness during transformation; merely fixing the initial seed does not make every repeated transformation identical because the random state advances.

### 7.3 Predictive averaging versus statistical pooling

For this exercise, multiple imputation means combining predictive probabilities. In statistical inference, multiple-imputation pooling also accounts for uncertainty within and between completed datasets, commonly through Rubin’s rules for estimated quantities. Averaging probabilities alone does not provide those inferential variance estimates.

Every ensemble member must still be evaluated with fold-specific fitting. Training on all development rows is appropriate for the final deployed models, but those training predictions must not replace out-of-fold performance estimates.

## 8. Comparing imputers fairly

The required comparison concerns the **complete predictive pipelines**, using `corrected_paired_test()` on validation scores from the same folds.

Keep the model, metric, split definitions, feature selection, encoder, scaler, and other settings consistent while changing the imputer. Tuning can be off. If tuning is enabled, it belongs inside the evaluation procedure, using nested CV as in earlier weeks. Keep the locked test set out of imputer selection.

For fold `i`, define the paired difference:

```text
d_i = score_A_i − score_B_i
```

For a metric where larger is better, a positive average difference favours A. Fold scores are correlated because training sets overlap, so a conventional independent-sample comparison underestimates uncertainty.

The project’s corrected paired test uses:

```text
SE_corrected = sqrt((1/k + n_validation/n_training) × sample_variance(d))
t = mean(d) / SE_corrected
```

It compares the statistic with a t distribution using `k − 1` degrees of freedom. In approximately equal-sized k-fold CV, the validation/training ratio is `1/(k − 1)`; with five folds it is 0.25. This ratio concerns CV folds, not the separate locked-test allocation.

Report the mean difference and p-value alongside fold scores and runtime. A large p-value does not prove equivalence, and a small p-value does not establish a practically useful gain. Several baseline comparisons also create multiple-testing concerns. If performance differences are weak, a faster, simpler imputer may be the sensible choice.

## 9. Saving and loading the complete fitted pipeline

### 9.1 What needs to be saved

Saving only the classifier omits the learned transformations that define its inputs. The complete fitted pipeline can contain:

- Training-derived scaling parameters and imputation statistics or models.
- Category mappings and encoding parameters.
- The selected-feature mask.
- The trained classifier.

`joblib.dump()` serialises this fitted state; `joblib.load()` restores it:

```python
joblib.dump(pipeline, "models/model.joblib")
loaded = joblib.load("models/model.joblib")
```

Save after the final refit on all development rows. The final refit creates the model for deployment; cross-validation supplies the performance estimate. Notebook 11’s saved pipeline includes preprocessing, feature selection, and the model.

If pipeline caching was used during training, `pipeline.set_params(memory=None)` removes the cache dependency before saving. Learned fitted parameters remain in place.

### 9.2 Training and inference have different jobs

| Training                          | Inference                                    |
| --------------------------------- | -------------------------------------------- |
| Requires outcome labels           | Requires no outcome labels                   |
| Fits preprocessing and classifier | Applies fitted preprocessing and classifier  |
| Uses CV to compare procedures     | Uses the selected saved procedure            |
| Produces a fitted model artifact  | Produces one prediction record per input row |

Calling `fit()` or `fit_transform()` on the new batch would change the preprocessing and invalidate the saved model’s input meaning. At inference, use `predict_proba()` and the transformations it invokes.

A serialised pipeline does not automatically include cleaning functions performed outside it, configuration files, or arbitrary helper source code. Those rules, imports, and matching feature-engineering steps must remain available and consistent. Use compatible library versions; the notebook asks for the same versions used when saving. Only load trusted artifacts because joblib/pickle loading can execute code.

For deterministic preprocessing, compare probabilities before saving and after loading with `np.allclose()`. Notebook 11 obtains `True`. Stochastic posterior imputers require a controlled random-state comparison rather than assuming repeated calls must be identical.

## 10. Predicting a new CSV correctly

The standalone prediction flow is:

```text
Read CSV and retain original IDs/order
    → apply training cleaning rules, preserving rows
    → reproduce feature engineering and missingness flags
    → construct label-free model inputs
    → load and apply the fitted pipeline
    → output ID, probability, and thresholded prediction
```

The output columns in the notebook are:

| Column        | Meaning                                                         |
| ------------- | --------------------------------------------------------------- |
| `id`          | Original identifier, preserving the connection to the input row |
| `probability` | Model probability for positive class 1                          |
| `prediction`  | 1 when probability is at least 0.5, otherwise 0                 |

A threshold of 0.5 is the exercise’s decision rule. It does not establish that false positives and false negatives have equal costs or that probabilities are calibrated. ROC-AUC measures ranking across thresholds and does not itself choose this decision threshold.

The notebook uses `predict_proba(X)[:, 1]`, appropriate for its class ordering `[0, 1]`. More general code should identify the positive-class column from `pipeline.classes_`.

### 10.1 The deliberately messy 20-row example

| Input condition                       | Required handling                                          |
| ------------------------------------- | ---------------------------------------------------------- |
| `priors_count = "?"`                  | Cleaning converts it to a gap; the saved imputer fills it. |
| Missing `age`                         | The saved imputer supplies an estimate.                    |
| Previously unseen charge degree `"X"` | The configured encoder uses its unknown-category policy.   |
| Duplicate person in rows 14 and 15    | Both rows receive an output, preserving order and IDs.     |

Unknown categories are distinct from missing values. An unfamiliar category can remain present and be handled by the encoder: one-hot encoding may map it to all zeros, ordinal encoding to a reserved code, and target encoding to a learned fallback. Avoiding a crash does not guarantee the model understands the new category accurately.

Training deduplication helps prevent repeated observations from distorting evaluation. Prediction has a different contract: **20 input rows must produce 20 output rows**, even if some represent the same person. Equal duplicate inputs produce equal outputs in the notebook’s deterministic pipeline; stochastic imputation need not preserve that equality on every draw.

The intended project-root commands, after implementing this week’s changes, are:

```bash
python main.py
python predict.py data/compas_new_people.csv predictions.csv
```

The first trains, evaluates, and saves. The second loads the saved model and predicts without retraining. The example `models/model.joblib` path is overwritten on subsequent training runs, so the loaded model must match the configuration and dataset used for prediction.

## 11. What needs to be done this week

**The notebooks require Part A and the README update by Saturday at 23:59.** Relative to the current project date of Friday 20 November 2026, that is Saturday **21 November 2026 at 23:59**; the notebook itself gives no calendar date or timezone. Part B is optional. Checked boxes below are supported by the current source files or saved project results; unchecked boxes still need work or verification. This review did not rerun training or inference.

### Required A1: make the imputer configurable and compare methods

- [x] Update `src/preprocessing.py` with `KNNImputer`, `IterativeImputer`, its enabling experimental import, and `RandomForestRegressor`.
- [x] Add `_NUMERIC_IMPUTERS` for `median`, `mean`, `knn`, `iterative`, and `random_forest`.
- [x] Change the numeric preprocessing branch to **scale → impute**, retaining categorical imputation/encoding and missingness flags.
- [x] Extend `preprocessing.imputation` in `config.yaml` with `numeric_strategy`, `categorical_strategy`, `n_neighbors`, `max_iter`, and `n_estimators`; retain a seed for model-based imputers (`preprocessing.random_state: 42`).
- [x] Run **median, KNN, iterative, and random-forest imputation** under comparable conditions. Tuning off is explicitly acceptable. All four, plus mean, are recorded in `results.md` and saved reports/CSV files; saved configurations match apart from `numeric_strategy`.
- [x] Compare the same-fold validation scores with `corrected_paired_test()`. Scores, mean differences, corrected t statistics and p-values are recorded in `results.md` and `validation/w10/imputer_comparison.json`; retain median because no alternative demonstrates a gain.
- [ ] Record training/evaluation runtimes for the imputer methods. The existing runs did not record elapsed time; this comparison did not retrain them.
- [x] Include the imputer setting in the console scoreboard, `results/runs.csv`, and explicit report header through `main.py` and `src/results.py`. New runs record `imputer` (defaulting to `median` if omitted); older rows without that field display `unknown` in the scoreboard.

**Comparison evidence:** the five saved runs use the same development split and five stratified CV folds, XGBoost, target encoding, robust scaling, voting selection with `min_votes=2`, engineered features off, and tuning off. `corrected_paired_test()` in `src/evaluate.py` has now been run on the per-fold `validation` column in the CSV files listed below with `test_train_ratio=0.25` (4 degrees of freedom, two-sided). Full-precision results and source scores are saved in `validation/w10/imputer_comparison.json`; `results.md` contains the comparison table. All four unadjusted p-values exceed 0.05. These results provide no evidence of a difference and do not prove equivalence. Runtime was not recorded in that summary or the run schema; timestamps alone do not establish elapsed runtime.

**Reproducibility follow-up:** the imputer seed is wired correctly, but the COMPAS target encoder currently uses `cv=5` without setting its random state. The existing `target_cv_random_state` option in `build_preprocessor()` can seed its shuffled folds. For a more tightly controlled comparison, set it consistently for every method and rerun all methods if using that change; do not mix seeded and earlier unseeded runs.

Example configuration fragment, merged into the existing `preprocessing` section:

```yaml
preprocessing:
  random_state: 42
  imputation:
    numeric_strategy: "median" # median | mean | knn | iterative | random_forest
    categorical_strategy: "most_frequent"
    n_neighbors: 5
    max_iter: 10
    n_estimators: 50
```

### Required A2: save the model and add independent prediction

- [ ] Save the **whole fitted pipeline** from `main.py`, after its final development-set refit. Clear its training-cache reference if applicable.
- [ ] Add `output.model_path: "models/model.joblib"` to `config.yaml`.
- [x] Add `joblib` explicitly to `requirements.txt`.
- [ ] Add `models/` and generated `predictions.csv` to `.gitignore`.
- [x] Create a standalone `predict.py` that loads the saved pipeline, applies consistent cleaning/features, and writes IDs, positive-class probabilities, and predictions. Supports `--config` for the configuration used during training.
- [ ] Copy `W10_Notebooks/data/compas_new_people.csv` into the project’s `data/` directory.
- [x] Implement reusable label-free input preparation and row-preserving inference logic: `split_features_target()` accepts a missing target, and the existing inference block in `main.py` uses `deduplicate=False`, checks order/IDs, recreates features, and identifies the positive-class probability column. This is existing Week 9 groundwork, not completion of standalone Week 10 prediction.
- [x] Confirm inference preserves input rows, order, IDs, and duplicates, handles missing/unseen values, and requires no target column. Verified `predict.py` with the notebook’s 20-row CSV, temporary saved pipelines and output files; also checked engineered features and positive-class lookup. No deployment model was retained.
- [ ] Run `python predict.py data/compas_new_people.csv predictions.csv` and verify **20 data rows**, excluding the CSV header.

Add the model path to the existing output settings rather than replacing them:

```yaml
output:
  results_dir: "results"
  model_path: "models/model.joblib"
```

**Adaptation notes for this repository:** the notebook helpers and the current project differ. The current `build_preprocessor()` accepts the full configuration and adds engineered numeric features when enabled; preserve that behaviour when incorporating the new branch. Its `clean_dataset()` deduplicates by default, so inference must explicitly pass `deduplicate=False`. Prediction must also recreate enabled engineered features before selecting inputs. The Week 9 inference block in `main.py` already demonstrates these row-preserving steps and positive-class lookup, but it predicts within a training run; A2 additionally requires persistence and a standalone loading script. The project supports both COMPAS and Adult configurations, so keep the saved artifact and prediction configuration aligned; the required 20-row example here is COMPAS.

### Required README update

- [ ] Add the **W10 row** to “Pipeline progress”.
- [ ] Under **Missing values**, report the imputer runs and corrected comparisons; explain in **2–3 sentences** which imputer is retained and why. Use actual project results rather than the notebook’s illustrative scores.
- [ ] Under **Predicting new data**, give the prediction command and explain the three output columns and one-output-per-input-row behaviour.
- [ ] Ensure Part A and the README are present in the fork by the stated deadline.

Actual project results from [results.md](../results.md), cross-checked against saved reports and per-fold CSVs:

| Imputer       | CV ROC-AUC mean ± SD | Difference versus median | Corrected p-value | Runtime    | Decision  |
| ------------- | -------------------- | ------------------------ | ----------------- | ---------- | --------- |
| Median        | 0.721 ± 0.012        | Baseline                 | —                 | Not recorded | Retain |
| Mean          | 0.719 ± 0.013        | -0.001905             | 0.531490      | Not recorded | No demonstrated gain |
| KNN           | 0.721 ± 0.013        | -0.000172             | 0.952530      | Not recorded | No demonstrated gain |
| Iterative     | 0.721 ± 0.013        | +0.000016             | 0.993483      | Not recorded | No demonstrated gain |
| Random forest | 0.720 ± 0.015        | -0.000862             | 0.712851      | Not recorded | No demonstrated gain |

The corresponding files are `results/Model_xgboost_<run>.txt` and `.csv`: median `20261120_201813`, mean `20261120_202307`, KNN `20261120_202347`, iterative `20261120_202425`, and random forest `20261120_202506`. The CSVs retain all five fold scores. The reported train–validation gaps are respectively +0.027, +0.029, +0.030, +0.032, and +0.033. Retain median: the corrected comparisons show no statistically significant difference from any alternative, and the more elaborate methods add complexity without a demonstrated ROC-AUC gain. Runtime superiority has not been measured.

Define the difference consistently as candidate minus median, so a positive ROC-AUC difference favours the candidate.

**Remaining required work, in order:** measure and record imputer runtimes, implement pipeline persistence and standalone prediction, verify the 20-row COMPAS example, then update the README and put the completed work in the fork. `predict.py` and an explicit `joblib` dependency have now been added. No deployment `models/` directory, COMPAS new-people CSV in the root `data/`, or Week 10 README entry was found during this review. `.gitignore` does not yet list `models/` or `predictions.csv`. Optional Part B does not block the required submission.

### Optional Part B challenges

| Challenge                   | What to do                                                                                                                                          | What to assess                                                                                                                     |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| B1: multiple imputation     | Train `m` pipelines with posterior-sampling iterative imputers and different seeds; average their probabilities.                                    | Does the predictive benefit justify the extra computation and storage?                                                             |
| B2: categorical information | Redesign preprocessing so model-based imputers can use categorical predictors with appropriate encoding and fold-specific fitting.                  | Do hidden-value errors and complete-pipeline validation scores improve?                                                            |
| B3: forest proximities      | Use `forest.apply(X)` to calculate the fraction of trees in which row pairs share a leaf; impute prior counts with proximity-weighted donor values. | How does reconstruction compare with forest iterative imputation, and can the procedure handle unseen people without their labels? |
| B4: safer prediction        | Validate required input columns, store library versions beside the model, and add a row-count preservation test.                                    | Are errors understandable and the model’s execution environment identifiable?                                                      |
