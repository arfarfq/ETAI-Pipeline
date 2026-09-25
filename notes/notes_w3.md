WEEK 3:

# EDA

Outputs:

- tested missingness verdict per column
- table of invalid values caught by domain rules
- duplicate check done two ways
- multicollinearity check
- summary **verdict table** that the Preprocessing notebook builds directly on.

**Where this goes next:** every finding here gets carried into `src/data_diagnostics.py` in the hands-on part of class - see the "From this notebook to the pipeline" section at the end.

You are free to apply these functions as part of your pipeline, as well as any other data exploration step you believe meaningfull.

## Data dictionary

| column            | type        | description                                                   | notable values                                                                     |
| ----------------- | ----------- | ------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `id`              | identifier  | internal record id                                            | not a model feature                                                                |
| `sex`             | categorical | defendant's sex                                               | `Male`, `Female`                                                                   |
| `age`             | numeric     | defendant's age (years) at screening                          |                                                                                    |
| `age_cat`         | categorical | age bucket                                                    | `Less than 25`, `25 - 45`, `Greater than 45`                                       |
| `race`            | categorical | defendant's race, as recorded                                 | excluded from model features- kept aside only to audit fairness afterward          |
| `juv_fel_count`   | numeric     | number of prior juvenile felony offenses                      |                                                                                    |
| `juv_misd_count`  | numeric     | number of prior juvenile misdemeanor offenses                 |                                                                                    |
| `juv_other_count` | numeric     | number of other prior juvenile offenses                       |                                                                                    |
| `juvenile_total`  | numeric     | total juvenile offenses                                       |                                                                                    |
| `priors_count`    | numeric     | number of prior adult offenses                                |                                                                                    |
| `prior_offenses`  | numeric     | number of prior offenses                                      |                                                                                    |
| `age_in_months`   | numeric     | age expressed in months                                       |                                                                                    |
| `c_charge_degree` | categorical | degree of the current charge                                  | `F` (felony), `M` (misdemeanor)                                                    |
| `decile_score`    | numeric     | COMPAS's own risk score                                       | 1 (lowest) to 10 (highest); excluded from model features, used only for comparison |
| `score_text`      | categorical | COMPAS's own risk category                                    | `Low`, `Medium`, `High`; excluded from model features                              |
| `two_year_recid`  | binary      | **our target** - was this person rearrested within two years? | `0` = no, `1` = yes                                                                |

## Problems

### Wrong data types

df_raw.dtypes

priors_count -> object needs to be int

prior_offenses - > object needs to be int

age_cat -> potentially improved???

**Note that there are some non-numeric tokens in priors_count and prior_offenses**

## Dealing With NA

round(df_raw.isna().sum()/len(df_raw)\*100,2)

Checking proportion of missing values per column

### Technique 1: Cramer's V

Cramers_v shows if there is a correlation with missing values and variables

- MCAR, missing completely at random: whether a cell is missing has nothing to do with anything -- not the hidden value, not any other column. The missing values are a random sample, and a simple fill costs only precision.
- MAR, missing at random: whether a cell is missing depends on other, observed columns, but not on the hidden value itself once those are accounted for. A fill informed by those columns is defensible.
- MNAR, missing not at random: whether a cell is missing depends on the hidden value itself, or on something never recorded. No fill is safe on its own, and an explicit "was missing" flag often beats erasing the pattern.

well under 0.1 is weak/scattered, the closer you get to 1, the stronger the association

### Technique 2: Domain Rules

- `.isna()` only catches values that are _already_ `NaN`.
- It says nothing about values that are technically present but **impossible**
- an age of 5, a COMPAS decile score of 15 (the scale tops out at 10)
- Those need a rule that knows something about what the column actually means, not a generic missing-value scan.

#### Look out for placeholders -> missing values in disguise

- Perhaps deal with placeholders first

## Category Hygiene

- Use catonical maps to insure categories are

## Duplicates

- deal with complete duplicates and ID duplicates
- check for duplicates

df_raw.duplicated().sum()

### Multicollinearity

- correlation heatmap shows that some features are the same feature twice and we should decide how to deal with this

- `juvenile_total` doesn't stand out here the same way, though - its correlation with any _single_ other column is well under 1.0. That's the gap VIF is for: `juvenile_total` isn't a copy of one column, it's the _sum_ of three (`juv_fel_count` + `juv_misd_count` + `juv_other_count`), a three-way relationship pairwise correlation can't see but a regression-based check can.

**VIF** (Variance Inflation Factor) measures how much the variance of an estimated regression coefficient is increased because of multicolinearity (correlation among predictor variables) in a regression model.

- VIF = 1: Predictor has no correlation with other variables.
- VIF < 5: Generally considered acceptable and low concern.
- VIF >5/10 : Indicates high colinearity.

## The Verdict Table

## To Do:

## From this notebook to the pipeline

**Where to add:** create `src/data_diagnostics.py`. **What goes in it** - three functions, generic (driven by `config.yaml`, not hardcoded to COMPAS column names):

| Function                                                           | What it does                                                                       | Pulled from                              |
| ------------------------------------------------------------------ | ---------------------------------------------------------------------------------- | ---------------------------------------- |
| `test_missingness_mechanism(df, target_col, candidate_predictors)` | chi-square + Cramér's V, returns the verdict table                                 | the "Technique 1" cells above, unchanged |
| `flag_invalid_values(df, rules)`                                   | applies a dict of `{column: predicate}` domain rules, converts violations to `NaN` | the "Technique 2" cells above, unchanged |
| `find_duplicates(df, id_column)`                                   | both duplicate checks, returned together                                           | the "Technique 3" cells above, unchanged |

**What changes in `config.yaml`:** a new `diagnostics` section, so the module stays generic instead of hardcoding COMPAS's column names:

```yaml
diagnostics:
  candidate_predictors:
    ["sex", "race", "age_cat", "c_charge_degree", "score_text"]
  validity_rules:
    age: "18 <= age <= 100"
    decile_score: "1 <= decile_score <= 10"
    juv_fel_count: "juv_fel_count >= 0"
    priors_count: "0 <= priors_count <= 60"
  id_column: "id"
```

## Challenge - what else can you find?

The verdict table above covers what today's three techniques were pointed at. It isn't necessarily everything wrong with this file. A few directions worth poking at, on your own or with a partner:

- Are there rows where two columns _disagree_ with each other - not individually invalid, but inconsistent _together_? (Hint: compare `age_cat` against the raw `age` value for the same row. Try the same idea between `score_text` and `decile_score`.)
- Is there a column here you'd want a domain rule for that this notebook didn't build one for?
- Is there any other invalid value that was not found in this notebook?
