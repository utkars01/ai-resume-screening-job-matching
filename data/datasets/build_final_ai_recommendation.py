import pandas as pd
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

RANKING_FILE = PROCESSED_DIR / "ranking_validation.csv"
GAP_FILE = PROCESSED_DIR / "prioritized_skill_gaps.csv"
RECOMMENDATION_FILE = (
    PROCESSED_DIR / "candidate_improvement_recommendations.csv"
)

OUTPUT_FILE = (
    PROCESSED_DIR / "final_ai_recommendation.csv"
)

TARGET_JOB_ID = "JOB_000131"
TARGET_RESUME_ID = "R_008066"


# ============================================================
# HELPERS
# ============================================================

def safe_float(value):
    if pd.isna(value):
        return 0.0
    return float(value)


def match_level(skill_coverage, semantic_similarity):
    """
    Human-readable assessment based on the two primary
    candidate-fit signals.

    This is NOT a hiring decision.
    """

    if skill_coverage >= 0.75 and semantic_similarity >= 0.65:
        return "HIGH"

    if skill_coverage >= 0.50 and semantic_similarity >= 0.50:
        return "MEDIUM"

    return "LOW"


def recommendation_text(match, skill_coverage, missing_count):
    if match == "HIGH":
        return (
            "The candidate shows strong alignment with the selected "
            "job based on the available matching signals. The profile "
            "can be considered for further recruiter review."
        )

    if match == "MEDIUM":
        return (
            "The candidate shows partial alignment with the selected "
            "job. Reviewing the missing skills and experience areas "
            "is recommended before making a hiring decision."
        )

    if missing_count > 0:
        return (
            "The candidate currently has limited direct alignment "
            "with the selected job. Improving the identified "
            "priority skills could strengthen suitability. "
            "The profile should be reviewed by a human recruiter "
            "rather than being automatically rejected."
        )

    return (
        "The candidate currently shows limited alignment with "
        "the selected job and should receive further human review."
    )


# ============================================================
# START
# ============================================================

print("=" * 70)
print("PHASE 7.4 - FINAL AI RECOMMENDATION ENGINE")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading ranking validation data...")
ranking = pd.read_csv(RANKING_FILE)

print("Loading prioritized skill gaps...")
gaps = pd.read_csv(GAP_FILE)

print("Loading candidate improvement recommendations...")
recommendations = pd.read_csv(RECOMMENDATION_FILE)


# ============================================================
# VALIDATE DATA
# ============================================================

required_ranking_columns = {
    "job_id",
    "resume_id",
    "semantic_similarity",
    "skill_coverage",
}

missing = required_ranking_columns - set(ranking.columns)

if missing:
    raise ValueError(
        f"Missing ranking columns: {missing}"
    )


# ============================================================
# SELECT TARGET CANDIDATE
# ============================================================

candidate_rows = ranking[
    (ranking["job_id"].astype(str) == TARGET_JOB_ID)
    & (ranking["resume_id"].astype(str) == TARGET_RESUME_ID)
]

if candidate_rows.empty:
    raise ValueError(
        f"No ranking record found for "
        f"{TARGET_JOB_ID} / {TARGET_RESUME_ID}"
    )

candidate = candidate_rows.iloc[0]


# ============================================================
# CORE SIGNALS
# ============================================================

semantic_similarity = safe_float(
    candidate["semantic_similarity"]
)

skill_coverage = safe_float(
    candidate["skill_coverage"]
)


# ============================================================
# SKILL GAPS
# ============================================================

candidate_gaps = gaps[
    (gaps["job_id"].astype(str) == TARGET_JOB_ID)
    & (gaps["resume_id"].astype(str) == TARGET_RESUME_ID)
].copy()

missing_skills = candidate_gaps["skill"].tolist()

matched_skill_count = max(
    int(round(skill_coverage * 8)),
    0
)

missing_skill_count = len(missing_skills)


# ============================================================
# PRIORITY RECOMMENDATIONS
# ============================================================

candidate_recommendations = recommendations[
    (recommendations["job_id"].astype(str) == TARGET_JOB_ID)
    & (
        recommendations["resume_id"].astype(str)
        == TARGET_RESUME_ID
    )
].copy()

candidate_recommendations = candidate_recommendations.sort_values(
    "priority_score",
    ascending=False,
)


# ============================================================
# MATCH LEVEL
# ============================================================

match = match_level(
    skill_coverage,
    semantic_similarity,
)


# ============================================================
# FINAL RECOMMENDATION
# ============================================================

final_recommendation = recommendation_text(
    match,
    skill_coverage,
    missing_skill_count,
)


# ============================================================
# TOP IMPROVEMENTS
# ============================================================

top_improvements = candidate_recommendations[
    candidate_recommendations["priority"]
    .isin(["HIGH", "MEDIUM"])
]["skill"].head(5).tolist()


# ============================================================
# BUILD FINAL RECORD
# ============================================================

result = {
    "job_id": TARGET_JOB_ID,
    "resume_id": TARGET_RESUME_ID,
    "match_level": match,
    "semantic_similarity": semantic_similarity,
    "skill_coverage": skill_coverage,
    "skill_gap_percentage": 1 - skill_coverage,
    "matched_skill_count": matched_skill_count,
    "missing_skill_count": missing_skill_count,
    "missing_skills": "|".join(missing_skills),
    "top_improvement_skills": "|".join(top_improvements),
    "final_recommendation": final_recommendation,
    "human_review_required": True,
}

result_df = pd.DataFrame([result])


# ============================================================
# SAVE
# ============================================================

result_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 70)
print("FINAL AI RECOMMENDATION")
print("=" * 70)

print(f"\nJob: {TARGET_JOB_ID}")
print(f"Candidate: {TARGET_RESUME_ID}")

print(f"\nMatch Level: {match}")

print(
    f"Semantic Similarity: "
    f"{semantic_similarity:.4f}"
)

print(
    f"Skill Coverage: "
    f"{skill_coverage:.2%}"
)

print(
    f"Skill Gap: "
    f"{(1 - skill_coverage):.2%}"
)

print(
    f"\nMissing Skills: "
    f"{missing_skill_count}"
)

if missing_skills:
    print("\nMissing Skills:")
    for skill in missing_skills:
        print(f"  ✗ {skill}")

print("\nTop Improvement Areas:")

if top_improvements:
    for skill in top_improvements:
        print(f"  → {skill}")
else:
    print("  None")

print("\nFinal Recommendation:")
print(final_recommendation)

print("\nHuman Review Required: YES")

print("\n" + "=" * 70)
print(f"Saved: {OUTPUT_FILE}")
print("=" * 70)