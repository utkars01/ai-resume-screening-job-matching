import pandas as pd
import ast
from pathlib import Path

JOB_FILE = Path("data/processed/job_postings_processed.csv")
SKILL_MASTER_FILE = Path("data/processed/skill_master.csv")
OUTPUT_FILE = Path("data/processed/job_skill_profiles.csv")


def parse_skills(value):
    if pd.isna(value):
        return []

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):
            return [
                str(skill).strip().lower()
                for skill in parsed
                if str(skill).strip()
            ]

    except (ValueError, SyntaxError):
        return []

    return []


print("Loading job dataset...")

jobs = pd.read_csv(JOB_FILE)

print(f"Jobs loaded: {len(jobs)}")


print("\nLoading skill master table...")

skill_master = pd.read_csv(SKILL_MASTER_FILE)

print(f"Skill master rows: {len(skill_master)}")


# Build normalized -> canonical mapping
skill_mapping = (
    skill_master[
        ["normalized_skill", "canonical_skill"]
    ]
    .drop_duplicates("normalized_skill")
    .set_index("normalized_skill")["canonical_skill"]
    .to_dict()
)

print(
    f"Normalized skill mappings: "
    f"{len(skill_mapping)}"
)


job_records = []

print("\nProcessing job skills...")

for _, row in jobs.iterrows():

    job_id = row["job_id"]

    skills = parse_skills(row["skills_required"])

    canonical_skills = []

    for skill in skills:

        canonical = skill_mapping.get(skill)

        if canonical:
            canonical_skills.append(canonical)

    # Remove duplicate skills within a job
    canonical_skills = sorted(
        set(canonical_skills)
    )

    job_records.append({
        "job_id": job_id,
        "required_skills": canonical_skills,
        "required_skill_count": len(canonical_skills)
    })


job_profiles = pd.DataFrame(job_records)


# Convert lists into a readable string for CSV storage
job_profiles["required_skills"] = job_profiles[
    "required_skills"
].apply(
    lambda skills: "|".join(skills)
)


# Save
OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

job_profiles.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 70)
print("JOB SKILL PROFILES CREATED")
print("=" * 70)

print(
    f"Total job profiles: "
    f"{len(job_profiles)}"
)

print(
    f"Jobs with at least one skill: "
    f"{(job_profiles['required_skill_count'] > 0).sum()}"
)

print(
    f"Jobs without extracted skills: "
    f"{(job_profiles['required_skill_count'] == 0).sum()}"
)


print("\nRequired skill count distribution:")

print(
    job_profiles["required_skill_count"]
    .describe()
)


print("\nSample job skill profiles:")

print(
    job_profiles.head(10).to_string(
        index=False
    )
)


print(f"\nSaved to: {OUTPUT_FILE}")