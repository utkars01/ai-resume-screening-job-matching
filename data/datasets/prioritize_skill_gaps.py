import pandas as pd
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

GAP_FILE = PROCESSED_DIR / "skill_gap_analysis.csv"
FREQUENCY_FILE = PROCESSED_DIR / "skill_frequency.csv"
MASTER_FILE = PROCESSED_DIR / "skill_master.csv"

OUTPUT_FILE = PROCESSED_DIR / "prioritized_skill_gaps.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("PHASE 7.2 - INTELLIGENT SKILL GAP PRIORITIZATION")
print("=" * 70)

print("\nLoading skill-gap analysis...")
gaps = pd.read_csv(GAP_FILE)

print("Loading skill frequency data...")
frequency = pd.read_csv(FREQUENCY_FILE)

print("Loading skill master...")
master = pd.read_csv(MASTER_FILE)

print(f"Gap records: {len(gaps):,}")
print(f"Frequency records: {len(frequency):,}")
print(f"Master records: {len(master):,}")


# ============================================================
# INSPECT COLUMNS
# ============================================================

print("\nFrequency columns:")
print(list(frequency.columns))

print("\nMaster columns:")
print(list(master.columns))


# ============================================================
# NORMALIZE SKILL NAMES
# ============================================================

gaps["skill"] = gaps["skill"].astype(str).str.strip().str.lower()

frequency["canonical_skill"] = (
    frequency["canonical_skill"]
    .astype(str)
    .str.strip()
    .str.lower()
)

master["canonical_skill"] = (
    master["canonical_skill"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# ============================================================
# MERGE FREQUENCY INFORMATION
# ============================================================

frequency_columns = [
    "canonical_skill",
    "jobs_with_skill",
    "job_percentage",
]

frequency_available = [
    column for column in frequency_columns
    if column in frequency.columns
]

if "canonical_skill" not in frequency_available:
    raise ValueError(
        "skill_frequency.csv does not contain 'canonical_skill'."
    )

gaps = gaps.merge(
    frequency[frequency_available],
    left_on="skill",
    right_on="canonical_skill",
    how="left",
)

gaps.drop(columns=["canonical_skill"], inplace=True, errors="ignore")


# ============================================================
# MERGE SKILL CATEGORY
# ============================================================

master_columns = [
    "canonical_skill",
    "category",
]

master_available = [
    column for column in master_columns
    if column in master.columns
]

if "canonical_skill" in master_available:
    gaps = gaps.merge(
        master[master_available].drop_duplicates(
            subset=["canonical_skill"]
        ),
        left_on="skill",
        right_on="canonical_skill",
        how="left",
    )

    gaps.drop(columns=["canonical_skill"], inplace=True, errors="ignore")


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

if "jobs_with_skill" in gaps.columns:
    gaps["jobs_with_skill"] = gaps["jobs_with_skill"].fillna(0)

if "job_percentage" in gaps.columns:
    gaps["job_percentage"] = gaps["job_percentage"].fillna(0)

if "category" in gaps.columns:
    gaps["category"] = gaps["category"].fillna("unknown")


# ============================================================
# CALCULATE IMPORTANCE SCORE
# ============================================================

if "job_percentage" in gaps.columns:
    max_percentage = gaps["job_percentage"].max()

    if max_percentage > 0:
        gaps["frequency_score"] = (
            gaps["job_percentage"] / max_percentage
        )
    else:
        gaps["frequency_score"] = 0.0
else:
    gaps["frequency_score"] = 0.0

# ============================================================
# PRIORITY CLASSIFICATION
# ============================================================

def classify_priority(score):
    if score >= 0.50:
        return "HIGH"
    elif score >= 0.15:
        return "MEDIUM"
    else:
        return "LOW"


gaps["priority_score"] = gaps["frequency_score"]

gaps["priority"] = gaps["priority_score"].apply(
    classify_priority
)


# ============================================================
# SORT
# ============================================================

gaps = gaps.sort_values(
    ["priority_score", "jobs_with_skill"],
    ascending=[False, False],
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

gaps.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 70)
print("PRIORITIZED SKILL GAPS")
print("=" * 70)

for _, row in gaps.iterrows():

    skill = row["skill"]
    priority = row["priority"]
    score = row["priority_score"]

    jobs_count = row.get("jobs_with_skill", 0)

    print(
        f"{priority:<8} | "
        f"{skill:<30} | "
        f"score={score:.3f} | "
        f"jobs={int(jobs_count)}"
    )


print("\n" + "=" * 70)

print("\nPriority distribution:")
print(
    gaps["priority"]
    .value_counts()
    .to_string()
)

print("\nSaved:")
print(OUTPUT_FILE)

print("=" * 70)