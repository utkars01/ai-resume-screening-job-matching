import pandas as pd
from pathlib import Path


MAPPING_FILE = Path(
    "data/processed/job_skills_canonical.csv"
)

FREQUENCY_FILE = Path(
    "data/processed/skill_frequency.csv"
)

OUTPUT_FILE = Path(
    "data/processed/skill_master.csv"
)


print("Loading canonical skill mappings...")

mapping = pd.read_csv(MAPPING_FILE)

print(
    f"Mappings loaded: {len(mapping)}"
)


print("\nLoading skill frequency data...")

frequency = pd.read_csv(FREQUENCY_FILE)

print(
    f"Frequency records loaded: {len(frequency)}"
)


# ---------------------------------------------------------
# Keep one row per original -> canonical mapping
# ---------------------------------------------------------

mapping = mapping[
    [
        "original_skill",
        "normalized_skill",
        "canonical_skill"
    ]
].drop_duplicates()


# ---------------------------------------------------------
# Merge frequency information
# ---------------------------------------------------------

master = mapping.merge(
    frequency[
        [
            "canonical_skill",
            "skill_frequency",
            "jobs_with_skill",
            "job_percentage"
        ]
    ],
    on="canonical_skill",
    how="left"
)


# ---------------------------------------------------------
# Add frequency category
# ---------------------------------------------------------

def frequency_category(value):

    if pd.isna(value):
        return "unknown"

    if value >= 1000:
        return "very_common"

    if value >= 100:
        return "common"

    if value >= 10:
        return "moderate"

    if value >= 3:
        return "rare"

    return "very_rare"


master["frequency_category"] = (
    master["jobs_with_skill"]
    .apply(frequency_category)
)


# ---------------------------------------------------------
# Sort by frequency
# ---------------------------------------------------------

master = master.sort_values(
    [
        "jobs_with_skill",
        "canonical_skill"
    ],
    ascending=[
        False,
        True
    ]
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

master.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 70)
print("SKILL MASTER TABLE CREATED")
print("=" * 70)

print(
    f"Total mapping rows: {len(master)}"
)

print(
    f"Unique canonical skills: "
    f"{master['canonical_skill'].nunique()}"
)


print("\nFrequency categories:")

print(
    master["frequency_category"]
    .value_counts()
)


print("\nTop 30 canonical skills:")

top = (
    master[
        [
            "canonical_skill",
            "jobs_with_skill",
            "job_percentage",
            "frequency_category"
        ]
    ]
    .drop_duplicates("canonical_skill")
    .head(30)
)

print(
    top.to_string(index=False)
)


print(
    f"\nSaved to: {OUTPUT_FILE}"
)