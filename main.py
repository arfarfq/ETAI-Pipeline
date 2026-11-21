"""
Entry point for the baseline predictive pipeline.

Run with:
    python main.py

This orchestrates development-set cross-validation, optional nested tuning,
fairness reporting, final refitting, optional inference, and result export.
The labelled test set remains locked.
"""
import yaml
from sklearn.pipeline import Pipeline
import optuna
from datetime import datetime
from tempfile import TemporaryDirectory
import joblib
from src.data import load_data
from src.preprocessing import split_features_target, split_dev_test, build_preprocessor, add_engineered_features
from src.model import build_model
from src.evaluate import fairness_report
from src.results import save_run, save_scores
from src.clean import clean_dataset
from sklearn.model_selection import StratifiedKFold
from src.evaluate import cross_validate_pipeline, cv_report, oof_classification_report, build_scorers
from src.tuning import tune_pipeline, nested_cross_validate, tuning_report
from src.model import build_pipeline
import argparse
import pandas as pd
import numpy as np
from pathlib import Path



def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():

    parser = argparse.ArgumentParser(description='Run the configured binary classification pipeline')
    parser.add_argument('config_path', nargs='?', default='config.yaml')
    args = parser.parse_args()

    config = load_config(path=args.config_path)

    model_type = config["model"]["type"]

    run = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load the raw data, clean it, and split into features and target
    df_raw = load_data(config["data"]["path"])

    df = clean_dataset(df=df_raw, diagnosis=config["diagnostics"])

    if config["engineered_features"] == True:
            df = add_engineered_features(df)

    X, y, extras = split_features_target(df, config["data"], config["preprocessing"]["mnar_indicator_sources"])

    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = split_dev_test(
    X, y, extras,
    test_size=config["test_set"]["size"],
    random_state=config["test_set"]["random_state"],
    )

    selection_config = config.get("feature_selection", {})
    with TemporaryDirectory(prefix="pipeline_cache_") as cache_dir:
        pipeline = build_pipeline(config, config["model"], selection_config,
                                memory=cache_dir)

        #############################################################################

        # Set up cross-validation and evaluate the model using the
        cv_config = config["cv"]

        # Shuffle is only relevant if n_splits > 1, but we allow it to be set in the config for consistency with other hyperparameters
        shuffle = cv_config.get("shuffle", True)

        # Set up cross-validation
        cv = StratifiedKFold(n_splits=cv_config["n_splits"], shuffle=shuffle,
                            random_state=cv_config.get("random_state") if shuffle else None)


        # Evaluate the model using cross-validation on the development set
        metrics = cv_config["metrics"]
        selection_metric = cv_config["selection_metric"]
        scorers = build_scorers(metrics)
        n_jobs = cv_config.get("n_jobs", 1)

        floor_pipeline = Pipeline([
            ("prep", build_preprocessor(config)),
            ("model", build_model({
                "type": "dummy",
                "params": {"dummy": {"strategy": "prior"}},
            })),
        ])


        floor_scores, _ = cross_validate_pipeline(floor_pipeline, X_dev, y_dev, cv,
                                                {selection_metric: scorers[selection_metric]}, selection_metric, n_jobs)
        fold_scores, oof_proba = cross_validate_pipeline(pipeline, X_dev, y_dev, cv, scorers, selection_metric, n_jobs)
        floor, untuned = floor_scores["validation"].mean(), fold_scores["validation"].mean()

        print(f"\nDefault hyperparameters ({args.config_path})")
        print(f"  {selection_metric}: {untuned:.3f} ± {fold_scores['validation'].std(ddof=1):.3f}   "
            f"train-validation gap: {fold_scores['gap'].mean():+.3f}   no-model floor: {floor:.3f}")
        report = (f"Hyperparameters from {args.config_path}: {config['model'].get('params')}\n\n"
                + cv_report(fold_scores, metrics, selection_metric, "Cross-validation")
                + f"\n\nNo-model floor (dummy, same folds): {selection_metric} = {floor:.3f}")
        final_scores = fold_scores

        tuning_config = config.get("tuning", {})
        tuning_enabled = tuning_config.get("enabled", False)

        if tuning_enabled:
            search_spaces = tuning_config.get("search_spaces") or {}

            if model_type not in search_spaces:
                raise ValueError(f"tuning is enabled but {args.config_path} has no tuning.search_spaces for "
                                f"'{model_type}'. Add one, or set tuning.enabled: false.")
            search_space = dict(search_spaces[model_type])
            if selection_config.get("enabled", False):
                search_space.update(selection_config.get("search_space", {}))

            n_trials = tuning_config["n_trials"]
            tuning_seed = tuning_config["random_state"]
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            inner_cv = StratifiedKFold(n_splits=tuning_config["n_splits"], shuffle=True, random_state=tuning_seed)

            # Honest estimate on the same outer folds; its out-of-fold probabilities feed the reports.
            print(f"\nTuning ({n_trials} trials, nested CV) ... this takes a while", flush=True)
            final_scores, oof_proba = nested_cross_validate(
                pipeline, X_dev, y_dev, cv, inner_cv, scorers, selection_metric,
                search_space, n_trials, tuning_seed, n_jobs=n_jobs,
            )
            print(f"  {selection_metric}: {final_scores['validation'].mean():.3f} ± "
                f"{final_scores['validation'].std(ddof=1):.3f}   (honest estimate of the tuned model)")
            report += "\n\n" + cv_report(final_scores, metrics, selection_metric,
                                        "Nested cross-validation (the tuning procedure, estimated honestly)")

            # The hyperparameters that are kept: the same procedure, once, on all development rows.
            pipeline, study = tune_pipeline(pipeline, X_dev, y_dev, inner_cv, scorers[selection_metric],
                                            search_space, n_trials, tuning_seed, n_jobs=n_jobs)
            print("  Chosen hyperparameters:")
            for name, value in study.best_params.items():
                print(f"    {name.replace('model__', ''):<20s} {value:.4g}" if isinstance(value, float)
                    else f"    {name.replace('model__', ''):<20s} {value}")
            report += "\n\n" + tuning_report(study, final_scores, selection_metric)

        print(f"\nAll metrics ({'tuned model' if tuning_enabled else 'default hyperparameters'}, mean ± std)")
        for name in metrics:
            print(f"  {name:<18s} {final_scores[name].mean():.3f} ± {final_scores[name].std(ddof=1):.3f}")
        report += "\n\n" + oof_classification_report(y_dev, oof_proba)
        report += "\n" + fairness_report(y_dev, oof_proba, extras_dev, sensitive_attr=config["data"]["sensitive_attr"])

        pipeline.fit(X_dev, y_dev)
        model_path = Path(
            config.get("output", {}).get("model_path", "models/model.joblib")
        )


        model_path.parent.mkdir(parents=True, exist_ok=True)
        pipeline.set_params(memory=None)  # Remove the temporary training-cache reference.
        joblib.dump(pipeline, model_path)

        inference_config = config.get("inference", {})
        if inference_config.get("enabled", False):
            new_raw = load_data(inference_config["path"])
            id_column = inference_config["id_column"]
            new_ids = new_raw[id_column].to_numpy(copy=True)

            # Apply the same cleaning, preserving every input row and its order.
            new_clean = clean_dataset(
                new_raw, config["diagnostics"], deduplicate=False
            )
            if not new_clean.index.equals(new_raw.index) or not new_clean[id_column].equals(new_raw[id_column]):
                raise ValueError("Inference cleaning must preserve every row, its order, and its ID.")
            if config["engineered_features"]:
                new_clean = add_engineered_features(new_clean)
            X_new, _, _ = split_features_target(
                new_clean, config["data"],
                config["preprocessing"]["mnar_indicator_sources"],
            )

            # Reuse fitted imputation, encoding, scaling, selection, and model.
            # Never call fit or fit_transform on the unseen people.
            positive_index = list(pipeline.classes_).index(1)
            probabilities = pipeline.predict_proba(X_new)[:, positive_index]
            if not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
                raise ValueError("Predicted probabilities must be finite and in [0, 1].")
            predictions = (probabilities >= 0.5).astype(int)
            output = pd.DataFrame({
                id_column: new_ids,
                "probability": probabilities,
                "prediction": predictions,
            })
            if len(output) != len(new_raw):
                raise ValueError("Every input row must receive a prediction.")
            output_path = Path(inference_config["output_path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output.to_csv(output_path, index=False)
            with output_path.with_suffix(".yaml").open("w") as handle:
                yaml.safe_dump({"run": run, "config_path": args.config_path,
                                "input_path": inference_config["path"], "rows": len(output),
                                "threshold": 0.5, "development_rows": len(X_dev)}, handle)
            print(f"Saved {len(output)} predictions to {output_path}")

        names = pipeline.named_steps["prep"].get_feature_names_out()
        if "select" in pipeline.named_steps:
            names = names[pipeline.named_steps["select"].get_support()]
        kept = "Features kept: " + ", ".join(names)
        print(kept)
        report += "\n" + kept + "\n"

        refit = f"Final model: {config["model"]["type"]} refit on all {len(X_dev)} development rows."
        report += "\n" + refit + "\n"

        row = {"run": run, "data": config["data"]["path"], "model": config["model"]["type"], "tuned": tuning_enabled,
            "encoder": config["preprocessing"]["encoder"], "scaler": config["preprocessing"]["scaler"],
            "imputer": config["preprocessing"].get("imputation", {}).get("numeric_strategy", "median"),
            "params": str(config["model"].get("params")), "tuned_params": str(study.best_params) if tuning_enabled else "",
            "metric": selection_metric, "mean": final_scores["validation"].mean(),
            "std": final_scores["validation"].std(ddof=1), "untuned": untuned, "floor": floor,
            **{f"{name}_mean": float(final_scores[name].mean()) for name in metrics},
            **{f"{name}_std": float(final_scores[name].std(ddof=1)) for name in metrics},
            **{f"fold_{i}": score for i, score in enumerate(final_scores["validation"], start=1)},
            "feature_selection": (f"vote(min_votes={pipeline.named_steps['select'].min_votes})"
                      if "select" in pipeline.named_steps else "off")}
        runs = save_scores(config.get("output", {}).get("scoreboard", "results/runs.csv"), row)
        same = runs[(runs["data"] == row["data"]) & (runs["metric"] == selection_metric)]
        board = (same.sort_values("mean", ascending=False)
                [["run", "model", "tuned", "feature_selection", "encoder", "scaler", "imputer", "mean", "std", "untuned", "floor"]]
                .fillna({"imputer": "unknown"})
                .rename(columns={"mean": selection_metric})
                .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(f"\nScoreboard (runs on the same data and metric; best first)\n{board}")
        report += f"\nScoreboard:\n{board}\n"

        path = save_run(config.get("output", {}).get("results_dir", "results"), run, config, report)
        final_scores.to_csv(Path(path).with_suffix(".csv"), index=False)
        print(f"\n{refit}\nFull report (per-fold tables, tuning, confusion matrix, fairness): {path}")

if __name__ == "__main__":

    main()
