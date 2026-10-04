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
from src.data import load_data
from src.preprocessing import preprocess, clean_dataset, drop_duplicates, split_features_target, split_dev_test, build_preprocessor
from src.model import build_model
from src.evaluate import evaluate, fairness_report
from src.results import save_run


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    df_raw = load_data(config["data"]["path"])

    df = clean_dataset(df=df_raw, diagnosis=config["diagnostics"])

    # training data only: the same person must not count twice, or sit in both dev and test
    df = drop_duplicates(df, config["diagnostics"]["id_column"])  

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


    pipeline.fit(X_dev, y_dev)

    # predict on both splits -- train accuracy vs. test accuracy is how we'll spot overfitting, not just how "good" the model looks
    y_train_pred = pipeline.predict(X_dev)
    y_test_pred = pipeline.predict(X_test)

    report = evaluate(y_dev, y_train_pred, y_test, y_test_pred)
    report += "\n" + fairness_report(
        y_test, y_test_pred, extras_test, sensitive_attr=config["data"]["sensitive_attr"]
    )

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")

if __name__ == "__main__":
    main()
