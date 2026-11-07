import pandas as pd
import yaml
from src.evaluate import corrected_paired_test

runs = pd.read_csv("results/runs.csv", dtype={"run": str})
on_id = "20261107_201349"
off_id = "20261107_201505"
a = runs.loc[runs["run"] == on_id]
b = runs.loc[runs["run"] == off_id]
assert len(a) == len(b) == 1, "Each ID must identify exactly one run"
a, b = a.iloc[0], b.iloc[0]
assert a["model"] == b["model"] == "xgboost"
assert str(a["tuned"]).lower() == str(b["tuned"]).lower() == "true"
assert a["metric"] == b["metric"] == "roc_auc"
for key in ["data", "encoder", "scaler"]:
    assert a[key] == b[key], key
with open("config.yaml") as handle:
    config = yaml.safe_load(handle)
k = config["cv"]["n_splits"]
cols = [f"fold_{i}" for i in range(1, k + 1)]
assert a[cols].notna().all() and b[cols].notna().all()
print(corrected_paired_test(a[cols], b[cols], 1 / (k - 1)))
