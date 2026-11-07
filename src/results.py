"""
Saving each run's results to disk.

Printing to the terminal is fine while you're watching it happen, but it's gone the moment you scroll past it or close the window. This module writes the full report (accuracy, classification report, fairness table) to a timestamped file in `results/` instead, so youcan open it again later, or compare two runs side by side after changing something in config.yaml.
"""
import os
from datetime import datetime
import pandas as pd

import yaml


def save_run(results_dir: str, run: str, config: dict, report_text: str) -> str:
    os.makedirs(results_dir, exist_ok=True)
    model_type = config["model"]["type"]
    path = os.path.join(results_dir, f"Model_{model_type}_{run}.txt")
    header = (
        f"Run: {run}\n"
        f"Model: {model_type}\n"
        f"Engineered features: {config['engineered_features']}\n"
        f"Test size: {config['test_set']['size']}  "
        f"random_state: {config['test_set']['random_state']}\n"
        f"Feature selection: {config.get('feature_selection', {'enabled': False})}\n"
        "Configuration:\n"
        + yaml.safe_dump(config, sort_keys=False)
        + "=" * 60 + "\n\n"
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(header + report_text)
    return path

def save_scores(path: str, row: dict) -> pd.DataFrame:
    """Appends one run to the CSV at `path` (created if needed) and returns all runs."""
    runs = pd.read_csv(path, dtype={"run": str}) if os.path.exists(path) else pd.DataFrame()
    runs = pd.concat([runs, pd.DataFrame([row])], ignore_index=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    runs.to_csv(path, index=False)
    return runs