import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path("data/processed/ranking_training_data.csv")

TRAIN_FILE = Path("data/processed/ranking_train.csv")
VALIDATION_FILE = Path("data/processed/ranking_validation.csv")
SUMMARY_FILE = Path("data/processed/ranking_split_summary.txt")

TEST_SIZE = 0.20
RANDOM_STATE = 42

# ============================================================
# LOAD DATA
# ============================================================

print("Loading ranking training data...")

df = pd.read_csv(INPUT_FILE)

print(f"Total rows: {len(df):,}")
print(f"Total jobs: {df['job_id'].nunique():,}")

# ============================================================
# CONVERT JOB IDS TO NORMAL PYTHON STRINGS
# ============================================================

job_ids = df["job_id"].astype(str).unique()

print(f"Unique job IDs: {len(job_ids):,}")

# ============================================================
# SPLIT JOB IDs
# ============================================================

train_jobs, validation_jobs = train_test_split(
    job_ids.tolist(),
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

train_jobs = set(train_jobs)
validation_jobs = set(validation_jobs)

print(f"Training jobs: {len(train_jobs):,}")
print(f"Validation jobs: {len(validation_jobs):,}")

# ============================================================
# CREATE DATASETS
# ============================================================

df["job_id"] = df["job_id"].astype(str)

train_df = df[df["job_id"].isin(train_jobs)].copy()
validation_df = df[df["job_id"].isin(validation_jobs)].copy()

# ============================================================
# VALIDATE NO JOB LEAKAGE
# ============================================================

overlap = set(train_df["job_id"]) & set(validation_df["job_id"])

if overlap:
    raise ValueError(
        f"Data leakage detected! {len(overlap)} jobs appear in both splits."
    )

print(f"Job overlap: {len(overlap)}")

# ============================================================
# SAVE
# ============================================================

train_df.to_csv(TRAIN_FILE, index=False)
validation_df.to_csv(VALIDATION_FILE, index=False)

# ============================================================
# SUMMARY
# ============================================================

summary = f"""
RANKING TRAIN / VALIDATION SPLIT
================================

Random state: {RANDOM_STATE}
Validation ratio: {TEST_SIZE}

TOTAL
-----
Rows: {len(df):,}
Jobs: {df['job_id'].nunique():,}
Resumes: {df['resume_id'].nunique():,}

TRAINING
--------
Rows: {len(train_df):,}
Jobs: {train_df['job_id'].nunique():,}
Resumes: {train_df['resume_id'].nunique():,}

VALIDATION
----------
Rows: {len(validation_df):,}
Jobs: {validation_df['job_id'].nunique():,}
Resumes: {validation_df['resume_id'].nunique():,}

JOB OVERLAP
-----------
{len(overlap)}

TRAIN LABEL DISTRIBUTION
------------------------
{train_df['relevance_label'].value_counts(normalize=True).sort_index().to_string()}

VALIDATION LABEL DISTRIBUTION
----------------------------
{validation_df['relevance_label'].value_counts(normalize=True).sort_index().to_string()}
"""

SUMMARY_FILE.write_text(summary, encoding="utf-8")

print(summary)

print("\nFiles created:")
print(TRAIN_FILE)
print(VALIDATION_FILE)
print(SUMMARY_FILE)

print("\nSTEP 6.5 COMPLETE")