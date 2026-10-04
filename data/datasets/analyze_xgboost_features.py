import pandas as pd
import xgboost as xgb
from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

MODEL_FILE = Path("models/xgboost_ranker.json")
FEATURE_FILE = Path("models/ranking_features.txt")

OUTPUT_CSV = Path("data/processed/xgboost_feature_importance.csv")
OUTPUT_TXT = Path("data/processed/xgboost_feature_importance.txt")
OUTPUT_PLOT = Path("data/processed/xgboost_feature_importance.png")

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading XGBoost model...")

model = xgb.Booster()
model.load_model(MODEL_FILE)

feature_columns = FEATURE_FILE.read_text(
    encoding="utf-8"
).splitlines()

print(f"Total features: {len(feature_columns)}")

# ============================================================
# GET FEATURE IMPORTANCE
# ============================================================

importance = model.get_score(
    importance_type="gain"
)

rows = []

for feature in feature_columns:
    rows.append({
        "feature": feature,
        "importance_gain": importance.get(feature, 0.0)
    })

importance_df = pd.DataFrame(rows)

# Normalize to percentage
total_gain = importance_df["importance_gain"].sum()

if total_gain > 0:
    importance_df["importance_percent"] = (
        importance_df["importance_gain"] / total_gain
    ) * 100
else:
    importance_df["importance_percent"] = 0.0

importance_df = importance_df.sort_values(
    "importance_gain",
    ascending=False
).reset_index(drop=True)

importance_df["rank"] = (
    importance_df.index + 1
)

importance_df = importance_df[
    [
        "rank",
        "feature",
        "importance_gain",
        "importance_percent"
    ]
]

# ============================================================
# DISPLAY
# ============================================================

print("\nXGBOOST FEATURE IMPORTANCE")
print("==========================\n")

print(
    importance_df.to_string(
        index=False,
        formatters={
            "importance_gain": "{:.6f}".format,
            "importance_percent": "{:.2f}%".format
        }
    )
)

# ============================================================
# SAVE CSV
# ============================================================

importance_df.to_csv(
    OUTPUT_CSV,
    index=False
)

# ============================================================
# SAVE TEXT REPORT
# ============================================================

report = """
XGBOOST FEATURE IMPORTANCE
==========================

Importance type: Gain

"""

report += importance_df.to_string(
    index=False,
    formatters={
        "importance_gain": "{:.6f}".format,
        "importance_percent": "{:.2f}%".format
    }
)

OUTPUT_TXT.write_text(
    report,
    encoding="utf-8"
)

# ============================================================
# CREATE PLOT
# ============================================================

plot_df = importance_df.head(15).sort_values(
    "importance_percent",
    ascending=True
)

plt.figure(figsize=(10, 7))

plt.barh(
    plot_df["feature"],
    plot_df["importance_percent"]
)

plt.xlabel("Importance (%)")
plt.ylabel("Feature")
plt.title("XGBoost Feature Importance - Top 15")

plt.tight_layout()

plt.savefig(
    OUTPUT_PLOT,
    dpi=150
)

plt.close()

# ============================================================
# COMPLETE
# ============================================================

print("\nFiles created:")
print(OUTPUT_CSV)
print(OUTPUT_TXT)
print(OUTPUT_PLOT)

print("\nSTEP 6.8 COMPLETE")