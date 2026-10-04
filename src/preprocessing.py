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
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder, OrdinalEncoder, TargetEncoder, StandardScaler, MinMaxScaler, RobustScaler,
)
from category_encoders import CountEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score


_SCALERS = {"none": "passthrough", "standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}


_ENCODERS = {
    "onehot": lambda seed: OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ),
    "ordinal": lambda seed: OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    ),
    "count": lambda seed: CountEncoder(
        handle_unknown=0,
        handle_missing=0
    ),
    "target": lambda seed: TargetEncoder(
        target_type="binary",
        cv=5
    ),
}


def find_placeholder_rows(series: pd.Series, tokens: set) -> pd.Series:
    return series.astype(str).str.strip().isin(tokens)


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

def add_missingness_indicators(df: pd.DataFrame, mnar_indicator_sources: list) -> pd.DataFrame:
    """Adds a `<col>_was_missing` flag for each MNAR-diagnosed column, before that
    column gets imputed -- so a model can still see the pattern even though the fill
    value itself (median/mode) can't carry it. Target-agnostic."""
    out = df.copy()
    for col in mnar_indicator_sources:
        if col in out.columns:
            out[f"{col}_was_missing"] = out[col].isna().astype(int)
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
    for col in ["priors_count", "prior_offenses"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col].replace(list(placeholder_tokens), np.nan), errors="coerce")

    # domain-rule violations -> NaN
    out = apply_domain_rules(out, diagnosis["domain_rules"])

    # category canonicalization (also folds placeholder tokens in race/sex/c_charge_degree/score_text to NaN)
    out = canonicalize_categories(out, diagnosis["canonical_maps"], placeholder_tokens)


    # redundant columns found via multicollinearity
    cols_to_drop = [c for c in diagnosis["columns_to_drop"] if c in out.columns and c != "id"]
    out = out.drop(columns=cols_to_drop)

    return out


def build_preprocessor(preprocessing_config: dict) -> ColumnTransformer:
    """
    Factory: builds a leak-safe ColumnTransformer for the chosen encoder/scaler pair --
    read from `config.yaml`'s `preprocessing` section (chosen there, not hardcoded
    here). Every encoder tolerates unseen
    categories at transform time. Nothing is fit here: fitting happens later, on the
    training part of each CV fold only, because this object is placed *inside* the
    model's sklearn Pipeline (see main.py).
    """
    encoder_name = preprocessing_config["encoder"]
    scaler_name = preprocessing_config["scaler"]
    numeric_features = preprocessing_config["numeric_features"]
    categorical_features = preprocessing_config["categorical_features"]
    mnar_indicator_sources = preprocessing_config.get("mnar_indicator_sources", [])
    imputation = preprocessing_config.get("imputation", {})

    scaler_factory = _SCALERS[scaler_name]
    scaler = scaler_factory() if callable(scaler_factory) else scaler_factory
    encoder = _ENCODERS[encoder_name](preprocessing_config.get("random_state"))

    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=imputation.get("numeric_strategy", "median"))),
        ("scale", scaler),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=imputation.get("categorical_strategy", "most_frequent"))),
        ("encode", encoder),
    ])

    indicator_cols = [f"{c}_was_missing" for c in mnar_indicator_sources]

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
        ("indicators", "passthrough", indicator_cols),
    ])


def split_features_target(df: pd.DataFrame, data_config: dict, mnar_indicator_sources: list):
    """
    Returns (X, y, extras). `y` is `None` and `extras` has no target column when called
    on label-free inference data -- nothing downstream requires the target to be present.
    """
    target = data_config["target"]
    sensitive_attr = data_config["sensitive_attr"]
    drop_columns = data_config.get("drop_columns", [])

    df = add_missingness_indicators(df, mnar_indicator_sources)
    y = df[target] if target in df.columns else None

    extras_cols = [c for c in [sensitive_attr, "score_text"] if c in df.columns]
    extras = df[extras_cols].copy() if extras_cols else None

    always_drop = set(drop_columns) | {target, sensitive_attr}
    feature_cols = [c for c in df.columns if c not in always_drop]
    X = df[feature_cols]
    return X, y, extras

def split_dev_test(X, y, extras, test_size: float, random_state: int):
    """
    Sets the final test set aside (week 4 -- replaces week 2/3's `split_train_test`).

    Stratified split of X, y and the extras frame (race/score_text, kept for the fairness
    report) together, so all three stay row-aligned. Returns a *development* set and a
    *locked test set*:
      - development set: everything we're allowed to learn from and compare models on.
        Cross-validation (src/evaluate.py) splits it again into train/validation folds.
      - locked test set: never used to fit, tune, compare or choose anything. Its size and seed live in config.yaml's `test_set` section and are never changed after today.
    """
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = train_test_split(
        X, y, extras, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_dev, X_test, y_dev, y_test, extras_dev, extras_test



def preprocess(
    df_raw: pd.DataFrame,
    config: dict,
    target: str,
    sensitive_attr: str,
    drop_columns: list,
    test_size: float,
    random_state: int,
):

    df = clean_dataset(df=df_raw, diagnosis=config["diagnostics"])

    # training data only: the same person must not count twice, or sit in both dev and test
    df = drop_duplicates(df, config["diagnostics"]["id_column"])  

    mnar_sources = config["preprocessing"]["mnar_indicator_sources"]

    X, y, extras = split_features_target(df, config["data"], mnar_sources)

    X_train, X_test, y_train, y_test, extras_test = split_dev_test(
        X, y, extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"],
    )

    preprocessor = build_preprocessor(config["preprocessing"])

    pipeline = Pipeline([
        ("prep", build_preprocessor(config["preprocessing"])),
        ("model", build_model(config["model"])),
    ])

    return X_train, X_test, y_train, y_test, extras_test
