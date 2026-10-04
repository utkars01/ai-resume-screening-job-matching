import pandas as pd
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

JOB_PROFILES = PROCESSED_DIR / "job_skill_profiles.csv"
RESUME_PROFILES = PROCESSED_DIR / "resume_skill_profiles.csv"
OUTPUT_FILE = PROCESSED_DIR / "skill_gap_analysis.csv"


# ============================================================
# HELPERS
# ============================================================

def parse_skill_string(value):
    """Convert pipe-separated skill string into a normalized set."""
    if pd.isna(value):
        return set()

    return {
        skill.strip().lower()
        for skill in str(value).split("|")
        if skill.strip()
    }


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("PHASE 7.1 - SKILL GAP ANALYSIS ENGINE")
print("=" * 70)

print("\nLoading job skill profiles...")
jobs = pd.read_csv(JOB_PROFILES)

print("Loading resume skill profiles...")
resumes = pd.read_csv(RESUME_PROFILES)

print(f"Jobs loaded: {len(jobs):,}")
print(f"Resumes loaded: {len(resumes):,}")


# ============================================================
# VALIDATE COLUMNS
# ============================================================

required_job_columns = {
    "job_id",
    "required_skills",
    "required_skill_count",
}

required_resume_columns = {
    "resume_id",
    "candidate_skills",
    "candidate_skill_count",
}

missing_job = required_job_columns - set(jobs.columns)
missing_resume = required_resume_columns - set(resumes.columns)

if missing_job:
    raise ValueError(f"Missing job columns: {missing_job}")

if missing_resume:
    raise ValueError(f"Missing resume columns: {missing_resume}")


# ============================================================
# SELECT A DEMONSTRATION JOB + CANDIDATE
# ============================================================

# Use the same example from the completed recruiter explanation.
TARGET_JOB_ID = "JOB_000131"
TARGET_RESUME_ID = "R_008066"

job_match = jobs[jobs["job_id"] == TARGET_JOB_ID]
resume_match = resumes[resumes["resume_id"] == TARGET_RESUME_ID]

if job_match.empty:
    raise ValueError(f"Job {TARGET_JOB_ID} not found.")

if resume_match.empty:
    raise ValueError(f"Resume {TARGET_RESUME_ID} not found.")

job = job_match.iloc[0]
resume = resume_match.iloc[0]


# ============================================================
# PARSE SKILLS
# ============================================================

required_skills = parse_skill_string(job["required_skills"])
candidate_skills = parse_skill_string(resume["candidate_skills"])

matched_skills = required_skills.intersection(candidate_skills)
missing_skills = required_skills - candidate_skills


# ============================================================
# CALCULATE GAP
# ============================================================

required_count = len(required_skills)
matched_count = len(matched_skills)
missing_count = len(missing_skills)

if required_count > 0:
    skill_coverage = matched_count / required_count
    skill_gap_percentage = (missing_count / required_count) * 100
else:
    skill_coverage = 0.0
    skill_gap_percentage = 0.0


# ============================================================
# GAP PRIORITY
# ============================================================

def assign_priority(skill):
    """
    Initial priority classification.

    This first version treats every missing required skill
    as a required gap. Later stages can incorporate skill
    frequency and job importance.
    """
    return "HIGH"


gap_records = []

for skill in sorted(missing_skills):
    gap_records.append(
        {
            "job_id": TARGET_JOB_ID,
            "resume_id": TARGET_RESUME_ID,
            "skill": skill,
            "priority": assign_priority(skill),
        }
    )


# ============================================================
# OUTPUT
# ============================================================

gap_df = pd.DataFrame(gap_records)

if gap_df.empty:
    gap_df = pd.DataFrame(
        columns=[
            "job_id",
            "resume_id",
            "skill",
            "priority",
        ]
    )

gap_df.to_csv(OUTPUT_FILE, index=False)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("SKILL GAP RESULT")
print("=" * 70)

print(f"\nJob: {TARGET_JOB_ID}")
print(f"Candidate: {TARGET_RESUME_ID}")

print(f"\nRequired skills : {required_count}")
print(f"Candidate skills: {len(candidate_skills)}")
print(f"Matched skills  : {matched_count}")
print(f"Missing skills  : {missing_count}")

print(f"\nSkill coverage : {skill_coverage:.2%}")
print(f"Skill gap      : {skill_gap_percentage:.2f}%")

print("\nMATCHED SKILLS:")
if matched_skills:
    for skill in sorted(matched_skills):
        print(f"  ✓ {skill}")
else:
    print("  None")

print("\nMISSING REQUIRED SKILLS:")
if missing_skills:
    for skill in sorted(missing_skills):
        print(f"  ✗ {skill}")
else:
    print("  None")

print("\n" + "=" * 70)
print(f"Saved: {OUTPUT_FILE}")
print("=" * 70)