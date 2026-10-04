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
FINAL_FILE = PROCESSED_DIR / "final_ai_recommendation.csv"

OUTPUT_FILE = (
    PROCESSED_DIR / "final_pipeline_validation.txt"
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("PHASE 7.5 - FINAL AI PIPELINE VALIDATION")
print("=" * 70)


# ============================================================
# LOAD FILES
# ============================================================

print("\nLoading ranking validation data...")
ranking = pd.read_csv(RANKING_FILE)

print("Loading skill-gap data...")
gaps = pd.read_csv(GAP_FILE)

print("Loading improvement recommendations...")
recommendations = pd.read_csv(RECOMMENDATION_FILE)

print("Loading final recommendation...")
final = pd.read_csv(FINAL_FILE)


# ============================================================
# BASIC VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("1. DATASET VALIDATION")
print("=" * 70)

print(f"Ranking records: {len(ranking):,}")
print(f"Gap records: {len(gaps):,}")
print(f"Recommendation records: {len(recommendations):,}")
print(f"Final recommendation records: {len(final):,}")


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_ranking = {
    "job_id",
    "resume_id",
    "semantic_similarity",
    "skill_coverage",
}

required_gap = {
    "job_id",
    "resume_id",
    "skill",
    "priority",
}

required_recommendation = {
    "job_id",
    "resume_id",
    "skill",
    "priority",
    "recommendation",
}

required_final = {
    "job_id",
    "resume_id",
    "match_level",
    "semantic_similarity",
    "skill_coverage",
    "skill_gap_percentage",
    "missing_skill_count",
    "final_recommendation",
    "human_review_required",
}


def check_columns(dataframe, required, name):
    missing = required - set(dataframe.columns)

    if missing:
        print(f"FAIL - {name}: missing {missing}")
        return False

    print(f"PASS - {name}: all required columns present")
    return True


column_results = []

column_results.append(
    check_columns(
        ranking,
        required_ranking,
        "Ranking data",
    )
)

column_results.append(
    check_columns(
        gaps,
        required_gap,
        "Skill-gap data",
    )
)

column_results.append(
    check_columns(
        recommendations,
        required_recommendation,
        "Recommendation data",
    )
)

column_results.append(
    check_columns(
        final,
        required_final,
        "Final recommendation",
    )
)


# ============================================================
# VALUE VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("2. VALUE VALIDATION")
print("=" * 70)

value_results = []


# Semantic similarity
semantic_valid = (
    ranking["semantic_similarity"].between(0, 1).all()
)

print(
    "PASS - Semantic similarity within [0, 1]"
    if semantic_valid
    else "FAIL - Invalid semantic similarity values"
)

value_results.append(semantic_valid)


# Skill coverage
coverage_valid = (
    ranking["skill_coverage"].between(0, 1).all()
)

print(
    "PASS - Skill coverage within [0, 1]"
    if coverage_valid
    else "FAIL - Invalid skill coverage values"
)

value_results.append(coverage_valid)


# Final recommendation coverage
final_coverage_valid = (
    final["skill_coverage"].between(0, 1).all()
).item()

print(
    "PASS - Final skill coverage valid"
    if final_coverage_valid
    else "FAIL - Final skill coverage invalid"
)

value_results.append(final_coverage_valid)


# Skill gap
gap_valid = (
    final["skill_gap_percentage"].between(0, 1).all()
).item()

print(
    "PASS - Final skill gap valid"
    if gap_valid
    else "FAIL - Final skill gap invalid"
)

value_results.append(gap_valid)


# Human review
review_valid = (
    final["human_review_required"]
    .astype(str)
    .str.upper()
    .isin(["TRUE", "YES"])
    .all()
)

print(
    "PASS - Human review requirement present"
    if review_valid
    else "FAIL - Human review requirement invalid"
)

value_results.append(review_valid)


# ============================================================
# CONSISTENCY CHECK
# ============================================================

print("\n" + "=" * 70)
print("3. CONSISTENCY VALIDATION")
print("=" * 70)

consistency_results = []


# Skill gap should equal 1 - coverage
calculated_gap = (
    1 - final["skill_coverage"]
)

gap_difference = (
    calculated_gap - final["skill_gap_percentage"]
).abs()

gap_consistent = (
    gap_difference < 0.000001
).all()

print(
    "PASS - Skill gap matches skill coverage"
    if gap_consistent
    else "FAIL - Skill gap mismatch"
)

consistency_results.append(gap_consistent)


# Match levels
valid_match_levels = {"LOW", "MEDIUM", "HIGH"}

match_level_valid = (
    set(final["match_level"].dropna().unique())
    .issubset(valid_match_levels)
)

print(
    "PASS - Match levels valid"
    if match_level_valid
    else "FAIL - Invalid match level"
)

consistency_results.append(match_level_valid)


# Recommendations exist
recommendation_exists = (
    final["final_recommendation"]
    .astype(str)
    .str.strip()
    .ne("")
    .all()
)

print(
    "PASS - Final recommendation generated"
    if recommendation_exists
    else "FAIL - Missing final recommendation"
)

consistency_results.append(recommendation_exists)


# ============================================================
# MULTI-CANDIDATE VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("4. MULTI-CANDIDATE VALIDATION")
print("=" * 70)

pairs = (
    ranking[
        ["job_id", "resume_id"]
    ]
    .drop_duplicates()
    .head(10)
)

print(
    f"Testing {len(pairs)} job/candidate combinations..."
)

multi_results = []

for _, pair in pairs.iterrows():

    job_id = pair["job_id"]
    resume_id = pair["resume_id"]

    ranking_match = ranking[
        (ranking["job_id"] == job_id)
        & (ranking["resume_id"] == resume_id)
    ]

    gap_match = gaps[
        (gaps["job_id"] == job_id)
        & (gaps["resume_id"] == resume_id)
    ]

    if ranking_match.empty:
        multi_results.append(False)
        continue

    semantic = float(
        ranking_match.iloc[0]["semantic_similarity"]
    )

    coverage = float(
        ranking_match.iloc[0]["skill_coverage"]
    )

    valid = (
        0 <= semantic <= 1
        and 0 <= coverage <= 1
    )

    multi_results.append(valid)

    print(
        f"{'PASS' if valid else 'FAIL'} | "
        f"{job_id} | {resume_id} | "
        f"semantic={semantic:.4f} | "
        f"coverage={coverage:.2%} | "
        f"gaps={len(gap_match)}"
    )


multi_valid = all(multi_results)

print(
    "\nPASS - Multi-candidate validation successful"
    if multi_valid
    else "\nFAIL - Multi-candidate validation found issues"
)


# ============================================================
# FINAL RESULT
# ============================================================

all_checks = (
    all(column_results)
    and all(value_results)
    and all(consistency_results)
    and multi_valid
)

print("\n" + "=" * 70)
print("FINAL VALIDATION RESULT")
print("=" * 70)

if all_checks:
    status = "PASS"
    print("\nPASS - COMPLETE AI PIPELINE VALIDATED")
else:
    status = "FAIL"
    print("\nFAIL - REVIEW VALIDATION ERRORS")


# ============================================================
# SAVE REPORT
# ============================================================

report_lines = [
    "PHASE 7.5 - FINAL AI PIPELINE VALIDATION",
    "=" * 70,
    "",
    f"Ranking records: {len(ranking):,}",
    f"Gap records: {len(gaps):,}",
    f"Recommendation records: {len(recommendations):,}",
    f"Final recommendation records: {len(final):,}",
    "",
    f"Column validation: {'PASS' if all(column_results) else 'FAIL'}",
    f"Value validation: {'PASS' if all(value_results) else 'FAIL'}",
    f"Consistency validation: {'PASS' if all(consistency_results) else 'FAIL'}",
    f"Multi-candidate validation: {'PASS' if multi_valid else 'FAIL'}",
    "",
    f"FINAL STATUS: {status}",
    "",
    "Important limitation:",
    "The current ranking labels are weak/synthetic supervision",
    "derived from structured rules rather than human recruiter",
    "judgments. Therefore validation confirms pipeline consistency",
    "and technical behavior, not real-world hiring accuracy.",
]

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as file:
    file.write("\n".join(report_lines))


print(f"\nValidation report saved:")
print(OUTPUT_FILE)

print("=" * 70)