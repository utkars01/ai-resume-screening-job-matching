import pandas as pd
import ast
import re
import unicodedata
from pathlib import Path


INPUT_FILE = Path(
    "data/processed/job_postings_processed.csv"
)

SKILL_MAPPING_FILE = Path(
    "data/processed/job_skills_canonical.csv"
)

OUTPUT_FILE = Path(
    "data/processed/skill_frequency.csv"
)


def normalize_text(value):
    if not isinstance(value, str):
        return ""

    value = unicodedata.normalize("NFKC", value)

    value = value.replace("\u2011", "-")
    value = value.replace("\u2013", "-")
    value = value.replace("\u2014", "-")

    value = re.sub(r"\s+", " ", value)

    return value.strip().lower()


print("Loading job dataset...")

jobs = pd.read_csv(INPUT_FILE)

print(f"Jobs loaded: {len(jobs)}")


print("\nLoading canonical skill mappings...")

mapping = pd.read_csv(SKILL_MAPPING_FILE)

print(
    f"Skill mappings loaded: {len(mapping)}"
)


# ---------------------------------------------------------
# Create original -> canonical mapping
# ---------------------------------------------------------

skill_map = dict(
    zip(
        mapping["normalized_skill"],
        mapping["canonical_skill"]
    )
)


# ---------------------------------------------------------
# Parse skills from every job
# ---------------------------------------------------------

skill_records = []

for index, value in enumerate(
    jobs["skills_required"]
):

    if pd.isna(value):
        continue

    try:
        parsed = ast.literal_eval(value)

    except (ValueError, SyntaxError):
        continue

    if not isinstance(parsed, list):
        continue

    for skill in parsed:

        normalized = normalize_text(skill)

        if not normalized:
            continue

        canonical = skill_map.get(
            normalized,
            normalized
        )

        skill_records.append({
            "job_index": index,
            "canonical_skill": canonical
        })


skills_df = pd.DataFrame(skill_records)


print(
    f"\nTotal job-skill occurrences: "
    f"{len(skills_df)}"
)


# ---------------------------------------------------------
# Frequency by occurrence
# ---------------------------------------------------------

frequency = (
    skills_df
    .groupby("canonical_skill")
    .size()
    .reset_index(name="skill_frequency")
)


# ---------------------------------------------------------
# Number of unique jobs containing each skill
# ---------------------------------------------------------

job_counts = (
    skills_df
    .groupby("canonical_skill")["job_index"]
    .nunique()
    .reset_index(name="jobs_with_skill")
)


frequency = frequency.merge(
    job_counts,
    on="canonical_skill"
)


# Percentage of jobs
total_jobs = len(jobs)

frequency["job_percentage"] = (
    frequency["jobs_with_skill"]
    / total_jobs
    * 100
)


# Sort
frequency = frequency.sort_values(
    "jobs_with_skill",
    ascending=False
)


# Save
frequency.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("REAL SKILL FREQUENCY ANALYSIS")
print("=" * 70)

print(
    f"Unique canonical skills: "
    f"{len(frequency)}"
)


print("\nTop 50 skills by number of jobs:")

print(
    frequency.head(50).to_string(
        index=False
    )
)


print("\nFrequency distribution:")

print(
    frequency["jobs_with_skill"].describe()
)


print("\nRare skills:")

rare = frequency[
    frequency["jobs_with_skill"] <= 2
]

print(
    f"Skills appearing in 2 jobs or fewer: "
    f"{len(rare)}"
)


print(
    f"\nSaved to: {OUTPUT_FILE}"
)