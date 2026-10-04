import pandas as pd
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_FILE = PROCESSED_DIR / "prioritized_skill_gaps.csv"
OUTPUT_FILE = PROCESSED_DIR / "candidate_improvement_recommendations.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("PHASE 7.3 - CANDIDATE IMPROVEMENT RECOMMENDATION ENGINE")
print("=" * 70)

print("\nLoading prioritized skill gaps...")

gaps = pd.read_csv(INPUT_FILE)

print(f"Gap records: {len(gaps):,}")


# ============================================================
# VALIDATION
# ============================================================

required_columns = {
    "job_id",
    "resume_id",
    "skill",
    "priority",
    "priority_score",
    "jobs_with_skill",
}

missing_columns = required_columns - set(gaps.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# RECOMMENDATION LOGIC
# ============================================================

def generate_recommendation(skill, priority, jobs_with_skill):
    """
    Generate a transparent recommendation based on
    skill priority and market frequency.
    """

    if priority == "HIGH":
        action = (
            f"Prioritize developing {skill}. "
            f"It appears frequently across the job dataset "
            f"({int(jobs_with_skill):,} jobs)."
        )

        focus = (
            f"Build practical knowledge of {skill}, "
            f"complete at least one relevant project or "
            f"practical exercise, and demonstrate it clearly "
            f"in the resume."
        )

    elif priority == "MEDIUM":
        action = (
            f"Consider improving {skill}. "
            f"It appears in {int(jobs_with_skill):,} jobs "
            f"in the available dataset."
        )

        focus = (
            f"Develop working-level knowledge of {skill} "
            f"through practical exercises or project work."
        )

    else:
        action = (
            f"Consider learning {skill} after higher-priority "
            f"skills have been addressed. It appears in "
            f"{int(jobs_with_skill):,} jobs."
        )

        focus = (
            f"Treat {skill} as a secondary development area "
            f"unless a target job specifically requires it."
        )

    return action, focus


# ============================================================
# BUILD RECOMMENDATIONS
# ============================================================

recommendations = []

for _, row in gaps.iterrows():

    skill = row["skill"]
    priority = row["priority"]
    jobs_with_skill = row["jobs_with_skill"]

    action, focus = generate_recommendation(
        skill,
        priority,
        jobs_with_skill,
    )

    recommendations.append(
        {
            "job_id": row["job_id"],
            "resume_id": row["resume_id"],
            "skill": skill,
            "priority": priority,
            "priority_score": row["priority_score"],
            "jobs_with_skill": int(jobs_with_skill),
            "recommendation": action,
            "development_focus": focus,
        }
    )


# ============================================================
# CREATE DATAFRAME
# ============================================================

recommendation_df = pd.DataFrame(recommendations)

recommendation_df = recommendation_df.sort_values(
    ["priority_score", "jobs_with_skill"],
    ascending=[False, False],
).reset_index(drop=True)


# ============================================================
# ADD ACTION ORDER
# ============================================================

recommendation_df["action_order"] = (
    range(1, len(recommendation_df) + 1)
)


# ============================================================
# SAVE
# ============================================================

recommendation_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 70)
print("CANDIDATE IMPROVEMENT RECOMMENDATIONS")
print("=" * 70)

for _, row in recommendation_df.iterrows():

    print(
        f"\n{row['action_order']}. "
        f"{row['priority']} - {row['skill']}"
    )

    print(
        f"   Job frequency: "
        f"{row['jobs_with_skill']:,} jobs"
    )

    print(
        f"   Recommendation: "
        f"{row['recommendation']}"
    )

    print(
        f"   Development focus: "
        f"{row['development_focus']}"
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("RECOMMENDATION SUMMARY")
print("=" * 70)

print(
    recommendation_df["priority"]
    .value_counts()
    .to_string()
)

print("\nSaved:")
print(OUTPUT_FILE)

print("=" * 70)