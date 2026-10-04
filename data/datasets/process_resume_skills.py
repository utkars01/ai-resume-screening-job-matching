import pandas as pd
import ast
import re
import unicodedata
from pathlib import Path

RESUME_FILE = Path("data/processed/resumes_processed.csv")
MASTER_FILE = Path("data/processed/skill_master.csv")
OUTPUT_FILE = Path("data/processed/resume_skills_normalized.csv")


def normalize_skill(skill):
    if not isinstance(skill, str):
        return ""

    skill = unicodedata.normalize("NFKC", skill)

    skill = skill.replace("\u2011", "-")
    skill = skill.replace("\u2013", "-")
    skill = skill.replace("\u2014", "-")

    skill = skill.strip()
    skill = re.sub(r"\s+", " ", skill)
    skill = skill.lower()

    return skill


def parse_list(value):
    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    # Handle Python/CSV list format
    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):
            return [
                str(item).strip()
                for item in parsed
                if str(item).strip()
            ]
    except (ValueError, SyntaxError):
        pass

    # Handle NumPy-style format:
    # ['Python' 'Java' 'Docker']
    items = re.findall(r"'([^']*)'", value)

    if items:
        return [
            item.strip()
            for item in items
            if item.strip()
        ]

    return []


print("Loading resume dataset...")

resumes = pd.read_csv(RESUME_FILE)

print(f"Resumes loaded: {len(resumes)}")


print("\nLoading skill master table...")

master = pd.read_csv(MASTER_FILE)

print(f"Skill mappings loaded: {len(master)}")


# Build normalized → canonical mapping
skill_mapping = (
    master[
        ["normalized_skill", "canonical_skill"]
    ]
    .drop_duplicates("normalized_skill")
    .set_index("normalized_skill")["canonical_skill"]
    .to_dict()
)


print(f"Normalized skill mappings: {len(skill_mapping)}")


resume_records = []


print("\nProcessing resume skills...")

for _, row in resumes.iterrows():

    resume_id = row["resume_id"]

    skills = parse_list(row["skills"])

    for skill in skills:

        normalized = normalize_skill(skill)

        if not normalized:
            continue

        canonical = skill_mapping.get(normalized)

        if canonical is None:
            canonical = normalized

        resume_records.append({
            "resume_id": resume_id,
            "original_skill": skill,
            "normalized_skill": normalized,
            "canonical_skill": canonical
        })


resume_skills = pd.DataFrame(resume_records)


# Remove duplicate skill entries
resume_skills = resume_skills.drop_duplicates(
    subset=[
        "resume_id",
        "canonical_skill"
    ]
)


# Save
OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

resume_skills.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 70)
print("RESUME SKILL INTELLIGENCE CREATED")
print("=" * 70)

print(f"Total resume-skill records: {len(resume_skills)}")

print(
    f"Unique resumes with skills: "
    f"{resume_skills['resume_id'].nunique()}"
)

print(
    f"Unique canonical resume skills: "
    f"{resume_skills['canonical_skill'].nunique()}"
)


print("\nTop 30 resume skills:")

top_skills = (
    resume_skills["canonical_skill"]
    .value_counts()
    .head(30)
)

print(top_skills.to_string())


print("\nSample mappings:")

sample = resume_skills[
    [
        "original_skill",
        "normalized_skill",
        "canonical_skill"
    ]
].drop_duplicates().head(30)

print(sample.to_string(index=False))


print(f"\nSaved to: {OUTPUT_FILE}")