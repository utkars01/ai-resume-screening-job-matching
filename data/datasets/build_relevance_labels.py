import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# JOB-RELATIVE WEAKLY SUPERVISED RELEVANCE LABEL GENERATION
# ============================================================

print("=" * 70)
print("JOB-RELATIVE RELEVANCE LABEL GENERATION")
print("=" * 70)


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "processed"

INPUT_FILE = PROCESSED_DIR / "ranking_features.csv"
OUTPUT_FILE = PROCESSED_DIR / "ranking_training_data.csv"


# ------------------------------------------------------------
# Load feature dataset
# ------------------------------------------------------------

print("\nLoading ranking features...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")


# ------------------------------------------------------------
# Validate required columns
# ------------------------------------------------------------

required_columns = [
    "job_id",
    "resume_id",
    "required_skill_match_ratio",
    "preferred_skill_match_ratio",
    "role_compatibility",
    "experience_compatibility",
    "seniority_compatibility",
    "education_compatibility",
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ------------------------------------------------------------
# Structured compatibility score
# ------------------------------------------------------------

print("\nCalculating structured compatibility score...")

# NOTE:
# semantic_similarity and baseline_score are intentionally
# excluded from the label calculation.

df["structured_compatibility_score"] = (
    0.40 * df["required_skill_match_ratio"]
    + 0.20 * df["preferred_skill_match_ratio"]
    + 0.15 * df["role_compatibility"]
    + 0.10 * df["experience_compatibility"]
    + 0.10 * df["seniority_compatibility"]
    + 0.05 * df["education_compatibility"]
)


# ------------------------------------------------------------
# Job-relative percentile
# ------------------------------------------------------------

print("\nCalculating job-relative compatibility percentiles...")

df["job_percentile"] = (
    df.groupby("job_id")["structured_compatibility_score"]
    .rank(method="average", pct=True)
)


# ------------------------------------------------------------
# Relevance labels
# ------------------------------------------------------------

print("\nGenerating relative relevance labels...")


def assign_relative_label(row):

    score = row["structured_compatibility_score"]
    percentile = row["job_percentile"]

    # Strong candidates:
    # Top 10% AND minimum structured compatibility.
    if percentile >= 0.90 and score >= 0.25:
        return 2

    # Moderate candidates:
    # Top 30% AND minimum compatibility.
    if percentile >= 0.70 and score >= 0.20:
        return 1

    return 0


df["relevance_label"] = (
    df.apply(assign_relative_label, axis=1)
)


# ------------------------------------------------------------
# Label distribution
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("RELEVANCE LABEL DISTRIBUTION")
print("=" * 70)

label_names = {
    0: "Not Relevant",
    1: "Partially Relevant",
    2: "Highly Relevant",
}

label_counts = (
    df["relevance_label"]
    .value_counts()
    .sort_index()
)

label_percentages = (
    df["relevance_label"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

for label in [0, 1, 2]:

    count = label_counts.get(label, 0)
    percentage = label_percentages.get(label, 0)

    print(
        f"{label} - {label_names[label]:20s}: "
        f"{count:>10,} ({percentage:>6.2f}%)"
    )


# ------------------------------------------------------------
# Job-level analysis
# ------------------------------------------------------------

print("\nChecking job-level label distribution...")

job_label_summary = (
    df.groupby("job_id")["relevance_label"]
    .value_counts()
    .unstack(fill_value=0)
)

for label in [0, 1, 2]:

    if label not in job_label_summary.columns:
        job_label_summary[label] = 0


jobs_with_high = (
    job_label_summary[2] > 0
).sum()

jobs_with_partial = (
    job_label_summary[1] > 0
).sum()

jobs_without_positive = (
    (job_label_summary[1] == 0)
    & (job_label_summary[2] == 0)
).sum()


print(
    f"Jobs with >=1 highly relevant candidate: "
    f"{jobs_with_high:,}"
)

print(
    f"Jobs with >=1 partially relevant candidate: "
    f"{jobs_with_partial:,}"
)

print(
    f"Jobs with no positive candidates: "
    f"{jobs_without_positive:,}"
)


# ------------------------------------------------------------
# Score statistics
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STRUCTURED COMPATIBILITY SCORE")
print("=" * 70)

print(
    df["structured_compatibility_score"]
    .describe()
    .round(4)
)


print("\n" + "=" * 70)
print("JOB PERCENTILE DISTRIBUTION")
print("=" * 70)

print(
    df["job_percentile"]
    .describe()
    .round(4)
)


# ------------------------------------------------------------
# Remove temporary columns
# ------------------------------------------------------------

final_columns = [
    col
    for col in df.columns
    if col not in [
        "structured_compatibility_score",
        "job_percentile"
    ]
]

training_data = df[final_columns].copy()


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("VALIDATING TRAINING DATA")
print("=" * 70)

print(
    f"Training rows    : {len(training_data):,}"
)

print(
    f"Training columns : {len(training_data.columns)}"
)

print(
    f"Jobs             : "
    f"{training_data['job_id'].nunique():,}"
)

print(
    f"Resumes          : "
    f"{training_data['resume_id'].nunique():,}"
)

print(
    f"Missing values   : "
    f"{training_data.isna().sum().sum():,}"
)

print(
    f"Label values     : "
    f"{sorted(training_data['relevance_label'].unique())}"
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

print("\nSaving training dataset...")

training_data.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved to:")
print(OUTPUT_FILE)


print("\n" + "=" * 70)
print("JOB-RELATIVE RELEVANCE LABEL GENERATION COMPLETE")
print("=" * 70)