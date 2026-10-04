import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

VALIDATION_FILE = Path("data/processed/ranking_validation.csv")
MODEL_FILE = Path("models/xgboost_ranker.json")
FEATURE_FILE = Path("models/ranking_features.txt")

OUTPUT_FILE = Path("data/processed/xgboost_ranking_evaluation.txt")

# ============================================================
# LOAD DATA
# ============================================================

print("Loading validation data...")

df = pd.read_csv(VALIDATION_FILE)

model = xgb.Booster()
model.load_model(MODEL_FILE)

feature_columns = FEATURE_FILE.read_text(
    encoding="utf-8"
).splitlines()

print(f"Validation rows: {len(df):,}")
print(f"Validation jobs: {df['job_id'].nunique():,}")
print(f"Features: {len(feature_columns)}")

# ============================================================
# PREPARE FEATURES
# ============================================================

X = df[feature_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

if X.isna().sum().sum() > 0:
    raise ValueError("Missing values found in validation features.")

# ============================================================
# GENERATE XGBOOST PREDICTIONS
# ============================================================

print("\nGenerating XGBoost ranking scores...")

dmatrix = xgb.DMatrix(X)

df["xgb_score"] = model.predict(dmatrix)

# ============================================================
# RANK CANDIDATES
# ============================================================

df["xgb_rank"] = (
    df.groupby("job_id")["xgb_score"]
    .rank(method="first", ascending=False)
)

df["baseline_rank"] = (
    df.groupby("job_id")["baseline_score"]
    .rank(method="first", ascending=False)
)

# ============================================================
# METRIC FUNCTIONS
# ============================================================

def dcg_at_k(labels, k):
    labels = np.asarray(labels)[:k]

    if len(labels) == 0:
        return 0.0

    discounts = np.log2(
        np.arange(2, len(labels) + 2)
    )

    gains = (2 ** labels) - 1

    return np.sum(gains / discounts)


def ndcg_at_k(labels, k):
    actual = dcg_at_k(labels, k)

    ideal_labels = sorted(
        labels,
        reverse=True
    )

    ideal = dcg_at_k(
        ideal_labels,
        k
    )

    if ideal == 0:
        return 0.0

    return actual / ideal


def reciprocal_rank(labels):
    for index, label in enumerate(labels, start=1):
        if label > 0:
            return 1.0 / index

    return 0.0


# ============================================================
# CALCULATE METRICS
# ============================================================

xgb_ndcg_values = []
baseline_ndcg_values = []

xgb_mrr_values = []
baseline_mrr_values = []

xgb_top1_values = []
baseline_top1_values = []

xgb_top5_values = []
baseline_top5_values = []

xgb_top10_values = []
baseline_top10_values = []

for job_id, group in df.groupby("job_id"):

    # XGBoost ranking
    xgb_group = group.sort_values(
        "xgb_score",
        ascending=False
    )

    xgb_labels = (
        xgb_group["relevance_label"]
        .astype(int)
        .tolist()
    )

    # Baseline ranking
    baseline_group = group.sort_values(
        "baseline_score",
        ascending=False
    )

    baseline_labels = (
        baseline_group["relevance_label"]
        .astype(int)
        .tolist()
    )

    # NDCG@10
    xgb_ndcg_values.append(
        ndcg_at_k(xgb_labels, 10)
    )

    baseline_ndcg_values.append(
        ndcg_at_k(baseline_labels, 10)
    )

    # MRR
    xgb_mrr_values.append(
        reciprocal_rank(xgb_labels)
    )

    baseline_mrr_values.append(
        reciprocal_rank(baseline_labels)
    )

    # Top 1
    xgb_top1_values.append(
        int(xgb_labels[0] > 0)
    )

    baseline_top1_values.append(
        int(baseline_labels[0] > 0)
    )

    # Top 5
    xgb_top5_values.append(
        int(any(label > 0 for label in xgb_labels[:5]))
    )

    baseline_top5_values.append(
        int(any(label > 0 for label in baseline_labels[:5]))
    )

    # Top 10
    xgb_top10_values.append(
        int(any(label > 0 for label in xgb_labels[:10]))
    )

    baseline_top10_values.append(
        int(any(label > 0 for label in baseline_labels[:10]))
    )

# ============================================================
# AGGREGATE
# ============================================================

xgb_ndcg = np.mean(xgb_ndcg_values)
baseline_ndcg = np.mean(baseline_ndcg_values)

xgb_mrr = np.mean(xgb_mrr_values)
baseline_mrr = np.mean(baseline_mrr_values)

xgb_top1 = np.mean(xgb_top1_values)
baseline_top1 = np.mean(baseline_top1_values)

xgb_top5 = np.mean(xgb_top5_values)
baseline_top5 = np.mean(baseline_top5_values)

xgb_top10 = np.mean(xgb_top10_values)
baseline_top10 = np.mean(baseline_top10_values)

# ============================================================
# IMPROVEMENT
# ============================================================

def improvement(new, old):
    if old == 0:
        return 0.0

    return ((new - old) / old) * 100


# ============================================================
# REPORT
# ============================================================

report = f"""
XGBOOST RANKING EVALUATION
==========================

Validation jobs: {df['job_id'].nunique():,}
Validation candidates: {len(df):,}

NDCG@10
-------

Baseline : {baseline_ndcg:.6f}
XGBoost  : {xgb_ndcg:.6f}
Improvement: {improvement(xgb_ndcg, baseline_ndcg):.2f}%

MRR
---

Baseline : {baseline_mrr:.6f}
XGBoost  : {xgb_mrr:.6f}
Improvement: {improvement(xgb_mrr, baseline_mrr):.2f}%

TOP-1 RELEVANT CANDIDATE
------------------------

Baseline : {baseline_top1:.4f}
XGBoost  : {xgb_top1:.4f}
Improvement: {improvement(xgb_top1, baseline_top1):.2f}%

TOP-5 RELEVANT CANDIDATE
------------------------

Baseline : {baseline_top5:.4f}
XGBoost  : {xgb_top5:.4f}
Improvement: {improvement(xgb_top5, baseline_top5):.2f}%

TOP-10 RELEVANT CANDIDATE
-------------------------

Baseline : {baseline_top10:.4f}
XGBoost  : {xgb_top10:.4f}
Improvement: {improvement(xgb_top10, baseline_top10):.2f}%
"""

print(report)

OUTPUT_FILE.write_text(
    report,
    encoding="utf-8"
)

print(f"Evaluation report saved to:")
print(OUTPUT_FILE)

print("\nSTEP 6.7 COMPLETE")