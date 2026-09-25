"""
Preprocessing -- deliberately minimal for week 2.

This is intentionally the weakest part of the pipeline:
    - missing values are simply dropped (no imputation strategy)
    - categorical columns are one-hot encoded with no thought given to unseen categories or cardinality
    - a single train/test split is used (no cross-validation)

You will replace this with something better in the coming weeks.

One thing that is NOT naive, on purpose: `sensitive_attr` (race) is kept out of the model's input features entirely. It's split alongside the data so it's still available afterwards -- not to train on, but to check whether the model treats different groups differently. See src/evaluate.py:fairness_report.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def find_placeholder_rows(series: pd.Series, tokens: set) -> pd.Series:
    return series.astype(str).str.strip().isin(tokens)


def canonicalize_categories(
    df: pd.DataFrame,
    columns_and_maps: dict,
    placeholder_tokens: set
) -> pd.DataFrame:

    out = df.copy()

    for col, mapping in columns_and_maps.items():
        if col not in out.columns:
            continue

        cleaned = out[col].astype("string").str.strip()
        lowered = cleaned.str.lower()

        out[col] = lowered.map(mapping).fillna(cleaned)

        # Convert placeholder strings to actual missing values
        out.loc[
            out[col].isin(placeholder_tokens),
            col
        ] = pd.NA

    return out

def apply_domain_rules(df, domain_rules):
    out = df.copy()

    for column, rule in domain_rules.items():
        if column not in out.columns:
            continue

        mask = out[column].notna()

        if "min" in rule:
            mask &= out[column] >= rule["min"]

        if "max" in rule:
            mask &= out[column] <= rule["max"]

        out.loc[~mask, column] = np.nan

    return out

def clean_dataset(df: pd.DataFrame, diagnosis: dict) -> pd.DataFrame:
    """
    Apply the EDA notebook's diagnosis: category cleanup, domain-rule / placeholder ->
    NaN conversion, de-duplication, redundant-column removal. Target-column-agnostic --
    safe to call on label-free inference data.
    """
    out = df.copy()
    placeholder_tokens = set(diagnosis["placeholder_tokens"])

    # numeric columns that loaded as text because of placeholder tokens
    for col in ["priors_count", "prior_offenses"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col].replace(list(placeholder_tokens), np.nan), errors="coerce")

    # domain-rule violations -> NaN
    out = apply_domain_rules(out, diagnosis["domain_rules"])

    # category canonicalization (also folds placeholder tokens in race/sex/c_charge_degree/score_text to NaN)
    out = canonicalize_categories(out, diagnosis["canonical_maps"], placeholder_tokens)

    # duplicates: exact row dupes and repeated ids point at the same rows here -- drop, keep first
    out = out.drop_duplicates()
    if "id" in out.columns:
        out = out.drop_duplicates(subset="id", keep="first")

    # redundant columns found via multicollinearity
    cols_to_drop = [c for c in diagnosis["columns_to_drop"] if c in out.columns and c != "id"]
    out = out.drop(columns=cols_to_drop)

    return out


def preprocess(
    df_raw: pd.DataFrame,
    diagnostics: dict,
    target: str,
    sensitive_attr: str,
    drop_columns: list,
    test_size: float,
    random_state: int,
):
    # naive: just drop rows with any missing values
    df = clean_dataset(df=df_raw, diagnosis=diagnostics)

    y = df[target]

    # kept aside for fairness auditing after training -- never used as a model input
    extras = df[[sensitive_attr, "score_text"]].copy()

    columns_to_exclude = [target, sensitive_attr] + [
        c for c in drop_columns if c in df.columns
    ]
    X = df.drop(columns=columns_to_exclude)

    # naive: one-hot encode all non-numeric columns, no further thought
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test, extras_train, extras_test = train_test_split(
        X, y, extras, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, extras_test
