"""
Entry point for the baseline predictive pipeline.

Run with:
    python main.py

This orchestrates the full (deliberately simple) pipeline:
    load config -> load data -> preprocess -> split -> train
    -> evaluate (train & test) -> save results
"""
import yaml
from sklearn.pipeline import Pipeline
import optuna

from src.data import load_data
from src.preprocessing_old import clean_dataset, split_features_target, split_dev_test, build_preprocessor
from src.model import build_model
from src.evaluate import fairness_report
from src.results import save_run
from src.clean import clean_dataset
from sklearn.model_selection import StratifiedKFold
from src.evaluate import cross_validate_pipeline, cv_report, oof_classification_report
from src.tuning import tune_pipeline, nested_cross_validate, tuning_report


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():

    config = load_config()


    # Load the raw data, clean it, and split into features and target
    df_raw = load_data(config["data"]["path"])

    df = clean_dataset(df=df_raw, diagnosis=config["diagnostics"])


    mnar_sources = config["preprocessing"]["mnar_indicator_sources"]
    X, y, extras = split_features_target(df, config["data"], mnar_sources)

    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = split_dev_test(
    X, y, extras,
    test_size=config["test_set"]["size"],
    random_state=config["test_set"]["random_state"],
    )

    pipeline = Pipeline([
        ("prep", build_preprocessor(config["preprocessing"])),
        ("model", build_model(config["model"])),
    ])

    # Set up cross-validation and evaluate the model using the
    cv_config = config["cv"]

    # Shuffle is only relevant if n_splits > 1, but we allow it to be set in the config for consistency with other hyperparameters
    shuffle = cv_config.get("shuffle", True)

    # Set up cross-validation 
    cv = StratifiedKFold(n_splits=cv_config["n_splits"], shuffle=shuffle,
                         random_state=cv_config.get("random_state") if shuffle else None)

    # Evaluate the model using cross-validation on the development set
    scoring = cv_config.get("scoring", "accuracy")
    n_jobs = cv_config.get("n_jobs", 1)

    # Cross-validation on the development set, to get an honest estimate of the model's performance before tuning
    fold_scores, y_oof = cross_validate_pipeline(pipeline, X_dev, y_dev, cv, scoring, n_jobs=n_jobs)


    header = f"Hyperparameters from config.yaml: {config['model'].get('params')}"
    print(header)
    report = header + "\n" + cv_report(fold_scores, scoring)

    tuning_config = config.get("tuning", {})


    tuning_enabled = tuning_config.get("enabled", False)


    if tuning_enabled:
        # Identify the model type to select its corresponding hyperparameter search space.
        model_type = config["model"]["type"]

        # Retrieve the configured search spaces, defaulting to an empty dictionary if none exist.
        search_spaces = tuning_config.get("search_spaces") or {}
        if model_type not in search_spaces:
            raise ValueError(f"tuning is enabled but config.yaml has no tuning.search_spaces for "
                            f"'{model_type}'. Add one, or set tuning.enabled: false.")
        search_space = search_spaces[model_type]

        # Set the number of optimization trials, random seed, and Optuna logging level.
        n_trials = tuning_config["n_trials"]
        tuning_seed = tuning_config["random_state"]
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Configure stratified inner cross-validation for hyperparameter tuning.
        inner_cv = StratifiedKFold(
            n_splits=tuning_config["n_splits"],
            shuffle=True,
            random_state=tuning_seed
        )

        # Estimate generalization performance using nested cross-validation.
        # Each outer fold evaluates a model tuned exclusively on its corresponding training data.
        # The resulting out-of-fold predictions are used to generate the evaluation reports.
        print(f"\nNested cross-validation: {cv.get_n_splits()} outer folds x {n_trials} trials x "
            f"{inner_cv.get_n_splits()} inner folds ...")
        nested_scores, y_oof = nested_cross_validate(
            pipeline, X_dev, y_dev, cv, inner_cv, scoring, search_space, n_trials, tuning_seed, n_jobs=n_jobs
        )
        report += "\n\nNested cross-validation (the tuning procedure, estimated honestly):\n"
        report += cv_report(nested_scores, scoring)

        # Tune the pipeline again using all development data to obtain the final hyperparameters.
        # This final tuning run is separate from nested cross-validation, which estimates performance.
        pipeline, study = tune_pipeline(
            pipeline, X_dev, y_dev, inner_cv, scoring, search_space, n_trials, tuning_seed, n_jobs=n_jobs
        )
        print()

        # Append the optimization results and nested cross-validation scores to the report.
        report += "\n\n" + tuning_report(study, nested_scores, scoring)


    report += "\n\n" + oof_classification_report(y_dev, y_oof)
    report += "\n" + fairness_report(y_dev, y_oof, extras_dev, sensitive_attr=config["data"]["sensitive_attr"])

    pipeline.fit(X_dev, y_dev)
    refit = f"Final model: {config['model']['type']} refit on all {len(X_dev)} development rows."
    if tuning_enabled:
        refit += f" Tuned hyperparameters: {study.best_params}"
    print(refit)
    report += "\n" + refit + "\n"

    locked = (f"Locked test set: {len(X_test)} rows set aside, not evaluated. "
              f"Development set: {len(X_dev)} rows.")
    print(locked)
    report += "\n" + locked + "\n"

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")

    


if __name__ == "__main__":
    main()
