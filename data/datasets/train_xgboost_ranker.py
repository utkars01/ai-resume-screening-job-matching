import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

TRAIN_FILE = Path("data/processed/ranking_train.csv")
VALIDATION_FILE = Path("data/processed/ranking_validation.csv")

MODEL_FILE = Path("models/xgboost_ranker.json")
FEATURE_FILE = Path("models/ranking_features.txt")

# ============================================================
# LOAD DATA
# ============================================================

print("Loading training data...")

train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)

print(f"Training rows: {len(train_df):,}")
print(f"Validation rows: {len(validation_df):,}")

# ============================================================
# FEATURES
# ============================================================

DROP_COLUMNS = [
    "job_id",
    "resume_id",
    "relevance_label",
    "rank_score_normalized"
]

FEATURE_COLUMNS = [
    col for col in train_df.columns
    if col not in DROP_COLUMNS
]

print("\nFeatures used:")
for feature in FEATURE_COLUMNS:
    print(" -", feature)

print(f"\nTotal features: {len(FEATURE_COLUMNS)}")

# ============================================================
# PREPARE X / Y
# ============================================================

X_train = train_df[FEATURE_COLUMNS].copy()
y_train = train_df["relevance_label"].astype(int)

X_validation = validation_df[FEATURE_COLUMNS].copy()
y_validation = validation_df["relevance_label"].astype(int)

# Convert everything to numeric
X_train = X_train.apply(pd.to_numeric, errors="coerce")
X_validation = X_validation.apply(pd.to_numeric, errors="coerce")

# Safety check
if X_train.isna().sum().sum() > 0:
    raise ValueError("Missing values found in training features.")

if X_validation.isna().sum().sum() > 0:
    raise ValueError("Missing values found in validation features.")

# ============================================================
# GROUP INFORMATION
# ============================================================

# XGBoost ranking requires the number of candidates
# belonging to each query/group.

train_groups = (
    train_df.groupby("job_id", sort=False)
    .size()
    .to_numpy()
)

validation_groups = (
    validation_df.groupby("job_id", sort=False)
    .size()
    .to_numpy()
)

print("\nRanking groups:")
print(f"Training jobs: {len(train_groups):,}")
print(f"Validation jobs: {len(validation_groups):,}")

print(
    f"Training candidates/job: "
    f"min={train_groups.min()}, "
    f"max={train_groups.max()}, "
    f"mean={train_groups.mean():.2f}"
)

print(
    f"Validation candidates/job: "
    f"min={validation_groups.min()}, "
    f"max={validation_groups.max()}, "
    f"mean={validation_groups.mean():.2f}"
)

# ============================================================
# CREATE XGBOOST RANKING DATASETS
# ============================================================

dtrain = xgb.QuantileDMatrix(
    X_train,
    label=y_train
)

dvalidation = xgb.QuantileDMatrix(
    X_validation,
    label=y_validation,
    ref=dtrain
)

# Set query/group sizes
dtrain.set_info(
    group=train_groups
)

dvalidation.set_info(
    group=validation_groups
)

# ============================================================
# MODEL PARAMETERS
# ============================================================

params = {
    "objective": "rank:ndcg",
    "eval_metric": "ndcg@10",
    "tree_method": "hist",
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "seed": 42
}

NUM_BOOST_ROUNDS = 300

# ============================================================
# TRAIN
# ============================================================

print("\nStarting XGBoost ranking training...")
print(f"Boosting rounds: {NUM_BOOST_ROUNDS}")

model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=NUM_BOOST_ROUNDS,
    evals=[
        (dtrain, "train"),
        (dvalidation, "validation")
    ],
    verbose_eval=25
)

# ============================================================
# SAVE MODEL
# ============================================================

MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

model.save_model(MODEL_FILE)

FEATURE_FILE.write_text(
    "\n".join(FEATURE_COLUMNS),
    encoding="utf-8"
)

print("\nModel saved:")
print(MODEL_FILE)

print("\nFeature list saved:")
print(FEATURE_FILE)

print("\nSTEP 6.6 COMPLETE")