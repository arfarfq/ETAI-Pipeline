
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder, OrdinalEncoder, TargetEncoder, StandardScaler, MinMaxScaler, RobustScaler,
)
from category_encoders import CountEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score



def canonicalize_categories(df: pd.DataFrame, columns_and_maps: dict, placeholder_tokens: set) -> pd.DataFrame:
    out = df.copy()
    for col, mapping in columns_and_maps.items():
        if col not in out.columns:
            continue
        cleaned = out[col].astype(str).str.strip()
        lowered = cleaned.str.lower()
        out[col] = lowered.map(mapping).fillna(cleaned)
        out.loc[out[col].astype(str).str.strip().isin(placeholder_tokens), col] = np.nan
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

def drop_duplicates(df: pd.DataFrame, id: list):
    out = df.drop_duplicates()
    if "id" in out.columns:
        out = out.drop_duplicates(subset="id", keep="first")
 
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
    for col in diagnosis["numeric_features"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col].replace(list(placeholder_tokens), np.nan), errors="coerce")

    # domain-rule violations -> NaN
    out = apply_domain_rules(out, diagnosis["domain_rules"])

    # category canonicalization (also folds placeholder tokens in race/sex/c_charge_degree/score_text to NaN)
    out = canonicalize_categories(out, diagnosis["canonical_maps"], placeholder_tokens)


    # redundant columns found via multicollinearity
    cols_to_drop = [c for c in diagnosis["columns_to_drop"] if c in out.columns and c != "id"]

    out = out.drop(columns=cols_to_drop)

    out = drop_duplicates(out, diagnosis["id_column"])


    return out