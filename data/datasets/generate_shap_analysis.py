import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

VALIDATION_FILE = Path(
    "data/processed/ranking_validation.csv"
)

MODEL_FILE = Path(
    "models/xgboost_ranker.json"
)

FEATURE_FILE = Path(
    "models/ranking_features.txt"
)

OUTPUT_FILE = Path(
    "data/processed/shap_feature_importance.csv"
)

# ============================================================
# LOAD MODEL AND DATA
# ============================================================

print("Loading XGBoost model...")

model = xgb.Booster()
model.load_model(MODEL_FILE)

df = pd.read_csv(VALIDATION_FILE)

feature_columns = FEATURE_FILE.read_text(
    encoding="utf-8"
).splitlines()

X = df[feature_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

print(f"Validation rows: {len(X):,}")
print(f"Features: {len(feature_columns)}")

# ============================================================
# SAMPLE DATA
# ============================================================

# SHAP can be computationally expensive.
# Use a representative validation sample.

SAMPLE_SIZE = min(5000, len(X))

X_sample = X.sample(
    n=SAMPLE_SIZE,
    random_state=42
)

print(f"SHAP sample size: {len(X_sample):,}")

# ============================================================
# CREATE SHAP EXPLAINER
# ============================================================

print("\nCreating SHAP TreeExplainer...")

explainer = shap.TreeExplainer(model)

# ============================================================
# CALCULATE SHAP VALUES
# ============================================================

print("Calculating SHAP values...")

shap_values = explainer.shap_values(X_sample)

# Handle possible SHAP output formats
if isinstance(shap_values, list):
    shap_values = shap_values[0]

shap_values = np.asarray(shap_values)

print(
    f"SHAP value shape: {shap_values.shape}"
)

# ============================================================
# GLOBAL FEATURE IMPORTANCE
# ============================================================

mean_abs_shap = np.abs(
    shap_values
).mean(axis=0)

shap_importance = pd.DataFrame({
    "feature": feature_columns,
    "mean_abs_shap": mean_abs_shap
})

shap_importance = shap_importance.sort_values(
    "mean_abs_shap",
    ascending=False
).reset_index(drop=True)

shap_importance["rank"] = (
    shap_importance.index + 1
)

total_importance = (
    shap_importance["mean_abs_shap"].sum()
)

if total_importance > 0:
    shap_importance["importance_percent"] = (
        shap_importance["mean_abs_shap"]
        / total_importance
    ) * 100
else:
    shap_importance["importance_percent"] = 0

shap_importance = shap_importance[
    [
        "rank",
        "feature",
        "mean_abs_shap",
        "importance_percent"
    ]
]

# ============================================================
# DISPLAY
# ============================================================

print("\nSHAP FEATURE IMPORTANCE")
print("=======================\n")

print(
    shap_importance.to_string(
        index=False,
        formatters={
            "mean_abs_shap": "{:.6f}".format,
            "importance_percent": "{:.2f}%".format
        }
    )
)

# ============================================================
# SAVE
# ============================================================

shap_importance.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSHAP report saved:")
print(OUTPUT_FILE)

print("\nSTEP 6.9.2 COMPLETE")