import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from backend.app.services.ranking_feature_contract import validate_ranking_labels

# ============================================================
# CONFIG
# ============================================================

DEFAULT_INPUT_FILE = Path("data/processed/ranking_training_data.csv")
DEFAULT_TRAIN_FILE = Path("data/processed/ranking_train.csv")
DEFAULT_VALIDATION_FILE = Path("data/processed/ranking_validation.csv")
DEFAULT_SUMMARY_FILE = Path("data/processed/ranking_split_summary.txt")

parser = argparse.ArgumentParser(
    description="Create a job-grouped ranking train/validation split."
)
parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_FILE)
parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_FILE)
parser.add_argument("--validation-output", type=Path, default=DEFAULT_VALIDATION_FILE)
parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_FILE)
args = parser.parse_args()
INPUT_FILE = args.input
TRAIN_FILE = args.train_output
VALIDATION_FILE = args.validation_output
SUMMARY_FILE = args.summary_output

TEST_SIZE = 0.20
RANDOM_STATE = 42

# ============================================================
# LOAD DATA
# ============================================================

print("Loading ranking training data...")

df = pd.read_csv(INPUT_FILE)

validate_ranking_labels(df, "Ranking source dataset")

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

validate_ranking_labels(train_df, "Training split")
validate_ranking_labels(validation_df, "Validation split")

print(f"Job overlap: {len(overlap)}")

# ============================================================
# SAVE
# ============================================================

TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
VALIDATION_FILE.parent.mkdir(parents=True, exist_ok=True)
SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
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
