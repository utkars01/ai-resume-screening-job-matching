import pandas as pd
from pathlib import Path

RESUME_FILE = Path("data/processed/resume_skill_profiles.csv")
JOB_FILE = Path("data/processed/job_skill_profiles.csv")
OUTPUT_FILE = Path("data/processed/skill_coverage_top_candidates.csv")

# Keep only the best candidates for each job
TOP_K = 50


def parse_skill_string(value):
    if pd.isna(value):
        return set()

    value = str(value).strip()

    if not value:
        return set()

    return {
        skill.strip()
        for skill in value.split("|")
        if skill.strip()
    }


print("=" * 70)
print("BUILDING TOP-K SKILL COVERAGE DATASET")
print("=" * 70)

# ---------------------------------------------------------
# 1. Load resume profiles
# ---------------------------------------------------------

print("\nLoading resume skill profiles...")

resumes = pd.read_csv(RESUME_FILE)

print(f"Resume profiles loaded: {len(resumes):,}")

resume_skill_lookup = {}

for _, row in resumes.iterrows():

    resume_id = row["resume_id"]

    skills = parse_skill_string(row["candidate_skills"])

    resume_skill_lookup[resume_id] = skills


print(f"Resume lookup created: {len(resume_skill_lookup):,}")


# ---------------------------------------------------------
# 2. Build skill → resume index
# ---------------------------------------------------------

print("\nBuilding skill index...")

skill_to_resumes = {}

for resume_id, skills in resume_skill_lookup.items():

    for skill in skills:

        skill_to_resumes.setdefault(skill, set()).add(resume_id)


print(f"Unique resume skills indexed: {len(skill_to_resumes):,}")


# ---------------------------------------------------------
# 3. Load jobs
# ---------------------------------------------------------

print("\nLoading job skill profiles...")

jobs = pd.read_csv(JOB_FILE)

print(f"Job profiles loaded: {len(jobs):,}")


# ---------------------------------------------------------
# 4. Find top candidates for each job
# ---------------------------------------------------------

print("\nFinding top candidates...")

coverage_records = []

processed_jobs = 0
jobs_with_matches = 0

for _, job in jobs.iterrows():

    job_id = job["job_id"]

    required_skills = parse_skill_string(
        job["required_skills"]
    )

    # Skip jobs without extracted skills
    if not required_skills:
        continue

    # Find candidates having at least one required skill
    candidate_resumes = set()

    for skill in required_skills:

        candidate_resumes.update(
            skill_to_resumes.get(skill, set())
        )

    if not candidate_resumes:
        continue

    jobs_with_matches += 1

    job_candidates = []

    for resume_id in candidate_resumes:

        candidate_skills = resume_skill_lookup[resume_id]

        matched = candidate_skills & required_skills

        missing = required_skills - candidate_skills

        matched_count = len(matched)

        required_count = len(required_skills)

        coverage = (
            matched_count / required_count
            if required_count > 0
            else 0
        )

        job_candidates.append({
            "resume_id": resume_id,
            "job_id": job_id,
            "matched_skills": "|".join(sorted(matched)),
            "missing_skills": "|".join(sorted(missing)),
            "matched_skill_count": matched_count,
            "required_skill_count": required_count,
            "skill_coverage": round(coverage, 4),
            "match_percentage": round(coverage * 100, 2)
        })

    # -----------------------------------------------------
    # Keep only TOP_K candidates for this job
    # -----------------------------------------------------

    job_candidates.sort(
        key=lambda x: (
            x["skill_coverage"],
            x["matched_skill_count"]
        ),
        reverse=True
    )

    coverage_records.extend(
        job_candidates[:TOP_K]
    )

    processed_jobs += 1

    if processed_jobs % 5000 == 0:

        print(
            f"Processed jobs: {processed_jobs:,} | "
            f"Records kept: {len(coverage_records):,}"
        )


# ---------------------------------------------------------
# 5. Save result
# ---------------------------------------------------------

coverage_df = pd.DataFrame(coverage_records)

if not coverage_df.empty:

    coverage_df = coverage_df.sort_values(
        [
            "job_id",
            "skill_coverage",
            "matched_skill_count"
        ],
        ascending=[True, False, False]
    )


OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

coverage_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TOP-K SKILL COVERAGE COMPLETE")
print("=" * 70)

print(
    f"Coverage records created: "
    f"{len(coverage_df):,}"
)

if not coverage_df.empty:

    print(
        f"Unique resumes represented: "
        f"{coverage_df['resume_id'].nunique():,}"
    )

    print(
        f"Unique jobs represented: "
        f"{coverage_df['job_id'].nunique():,}"
    )

    print("\nSkill coverage statistics:")

    print(
        coverage_df["skill_coverage"].describe()
    )

    print("\nTop 10 matches:")

    print(
        coverage_df[
            [
                "resume_id",
                "job_id",
                "matched_skills",
                "missing_skills",
                "matched_skill_count",
                "required_skill_count",
                "skill_coverage",
                "match_percentage"
            ]
        ].head(10).to_string(index=False)
    )


print(
    f"\nJobs with at least one candidate match: "
    f"{jobs_with_matches:,}"
)

print(
    f"\nSaved to: {OUTPUT_FILE}"
)