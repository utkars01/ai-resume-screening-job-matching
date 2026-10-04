import pandas as pd
import numpy as np
import ast
import re
from pathlib import Path


# ============================================================
# RANKING FEATURE ENGINEERING
# ============================================================

print("=" * 70)
print("RANKING FEATURE ENGINEERING")
print("=" * 70)


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "processed"

MATCHING_FILE = PROCESSED_DIR / "baseline_matching_results.csv"
RESUME_FILE = PROCESSED_DIR / "resumes_processed.csv"
JOB_FILE = PROCESSED_DIR / "job_postings_processed.csv"
OUTPUT_FILE = PROCESSED_DIR / "ranking_features.csv"


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def parse_list(value):
    """Safely convert stored list/string values into Python lists."""

    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    value = str(value).strip()

    if not value:
        return []

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):
            return [str(x).strip().lower() for x in parsed if str(x).strip()]

    except (ValueError, SyntaxError):
        pass

    # Fallback for comma-separated values
    return [
        x.strip().lower()
        for x in value.split(",")
        if x.strip()
    ]


def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"[^a-z0-9+#.\-/ ]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_role(role):
    """
    Normalize common role wording for lightweight
    title/role compatibility.
    """

    text = normalize_text(role)

    replacements = {
        "software engineer": "software engineer",
        "software developer": "software engineer",
        "application developer": "software engineer",
        "web developer": "web developer",
        "data scientist": "data scientist",
        "data analyst": "data analyst",
        "machine learning engineer": "machine learning",
        "ml engineer": "machine learning",
        "ai engineer": "artificial intelligence",
        "artificial intelligence engineer": "artificial intelligence",
        "devops engineer": "devops",
        "cloud engineer": "cloud",
        "backend developer": "backend",
        "backend engineer": "backend",
        "frontend developer": "frontend",
        "frontend engineer": "frontend",
    }

    return replacements.get(text, text)


def role_compatibility(resume_role, job_title):
    """
    Simple explainable role compatibility score.

    1.0 = strong direct compatibility
    0.5 = partial compatibility
    0.0 = no detected compatibility
    """

    resume_role = normalize_role(resume_role)
    job_title = normalize_text(job_title)

    if not resume_role or not job_title:
        return 0.0

    if resume_role in job_title:
        return 1.0

    resume_words = set(resume_role.split())
    job_words = set(job_title.split())

    if not resume_words:
        return 0.0

    overlap = len(resume_words & job_words) / len(resume_words)

    if overlap >= 0.5:
        return 0.75

    if overlap > 0:
        return 0.5

    return 0.0


def experience_compatibility(resume_years, required_years):
    """
    Compare candidate experience against job requirement.

    1.0 = requirement met
    Partial score when slightly below requirement.
    """

    try:
        resume_years = float(resume_years)
    except (TypeError, ValueError):
        return 0.5

    try:
        required_years = float(required_years)
    except (TypeError, ValueError):
        return 0.5

    if pd.isna(resume_years) or pd.isna(required_years):
        return 0.5

    if required_years <= 0:
        return 1.0

    if resume_years >= required_years:
        return 1.0

    ratio = resume_years / required_years

    if ratio >= 0.75:
        return 0.75

    if ratio >= 0.50:
        return 0.50

    return 0.25


def seniority_compatibility(resume_seniority, job_experience_level):
    """
    Lightweight seniority compatibility.
    """

    resume_level = normalize_text(resume_seniority)
    job_level = normalize_text(job_experience_level)

    if not resume_level or not job_level:
        return 0.5

    if resume_level == job_level:
        return 1.0

    pairs = {
        ("entry", "junior"),
        ("junior", "entry"),
        ("mid", "mid-level"),
        ("mid-level", "mid"),
        ("senior", "lead"),
        ("lead", "senior"),
    }

    if (resume_level, job_level) in pairs:
        return 0.75

    return 0.25


def education_compatibility(resume_education, job_education):
    """
    Lightweight education compatibility.
    """

    resume_text = normalize_text(resume_education)
    job_text = normalize_text(job_education)

    if not resume_text or not job_text:
        return 0.5

    if job_text in resume_text or resume_text in job_text:
        return 1.0

    # Degree keyword overlap
    degree_keywords = [
        "bachelor",
        "master",
        "phd",
        "btech",
        "mtech",
        "mba",
        "bca",
        "mca",
        "computer science",
        "engineering",
        "information technology",
    ]

    resume_keywords = {
        x for x in degree_keywords
        if x in resume_text
    }

    job_keywords = {
        x for x in degree_keywords
        if x in job_text
    }

    if resume_keywords & job_keywords:
        return 0.75

    return 0.25


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

print("\nLoading baseline matching results...")
matching = pd.read_csv(MATCHING_FILE)

print(f"Matching rows: {len(matching):,}")


print("\nLoading resume data...")
resumes = pd.read_csv(RESUME_FILE)

print(f"Resumes: {len(resumes):,}")


print("\nLoading job data...")
jobs = pd.read_csv(JOB_FILE)

print(f"Jobs: {len(jobs):,}")


# ------------------------------------------------------------
# Select useful resume columns
# ------------------------------------------------------------

resume_columns = [
    "resume_id",
    "role",
    "seniority",
    "years_experience",
    "education",
    "skills"
]

resume_columns = [
    col for col in resume_columns
    if col in resumes.columns
]

resumes_small = resumes[resume_columns].copy()


# ------------------------------------------------------------
# Select useful job columns
# ------------------------------------------------------------

job_columns = [
    "job_id",
    "title",
    "normalized_title",
    "experience_level",
    "years_experience_numeric",
    "education_level",
    "skills_required",
    "preferred_qualifications"
]

job_columns = [
    col for col in job_columns
    if col in jobs.columns
]

jobs_small = jobs[job_columns].copy()


# ------------------------------------------------------------
# Merge resume information
# ------------------------------------------------------------

print("\nMerging resume information...")

features = matching.merge(
    resumes_small,
    on="resume_id",
    how="left"
)

print(f"Rows after resume merge: {len(features):,}")


# ------------------------------------------------------------
# Merge job information
# ------------------------------------------------------------

print("\nMerging job information...")

features = features.merge(
    jobs_small,
    on="job_id",
    how="left"
)

print(f"Rows after job merge: {len(features):,}")


# ------------------------------------------------------------
# Parse skills
# ------------------------------------------------------------

print("\nParsing skills...")

features["resume_skills_list"] = features["skills"].apply(parse_list)

features["required_skills_list"] = features["skills_required"].apply(
    parse_list
)

features["preferred_skills_list"] = features[
    "preferred_qualifications"
].apply(parse_list)


# ------------------------------------------------------------
# Skill-based features
# ------------------------------------------------------------

print("\nCalculating skill features...")

def calculate_skill_features(row):

    resume_skills = set(row["resume_skills_list"])
    required_skills = set(row["required_skills_list"])
    preferred_skills = set(row["preferred_skills_list"])

    required_matched = resume_skills & required_skills
    preferred_matched = resume_skills & preferred_skills

    required_count = len(required_skills)
    preferred_count = len(preferred_skills)

    matched_required_count = len(required_matched)
    matched_preferred_count = len(preferred_matched)

    if required_count > 0:
        required_match_ratio = (
            matched_required_count / required_count
        )
    else:
        required_match_ratio = 0.0

    if preferred_count > 0:
        preferred_match_ratio = (
            matched_preferred_count / preferred_count
        )
    else:
        preferred_match_ratio = 0.0

    return pd.Series({
        "required_skill_count": required_count,
        "matched_required_skill_count": matched_required_count,
        "missing_required_skill_count":
            max(required_count - matched_required_count, 0),
        "required_skill_match_ratio": required_match_ratio,

        "preferred_skill_count": preferred_count,
        "matched_preferred_skill_count": matched_preferred_count,
        "preferred_skill_match_ratio": preferred_match_ratio,

        "resume_skill_count": len(resume_skills),

        "skill_union_count":
            len(resume_skills | required_skills),

        "skill_intersection_count":
            len(resume_skills & required_skills),
    })


skill_features = features.apply(
    calculate_skill_features,
    axis=1
)

features = pd.concat(
    [features, skill_features],
    axis=1
)


# ------------------------------------------------------------
# Role compatibility
# ------------------------------------------------------------

print("\nCalculating role compatibility...")

features["role_compatibility"] = features.apply(
    lambda row: role_compatibility(
        row.get("role", ""),
        row.get("normalized_title", row.get("title", ""))
    ),
    axis=1
)


# ------------------------------------------------------------
# Experience compatibility
# ------------------------------------------------------------

print("\nCalculating experience compatibility...")

features["experience_compatibility"] = features.apply(
    lambda row: experience_compatibility(
        row.get("years_experience", np.nan),
        row.get("years_experience_numeric", np.nan)
    ),
    axis=1
)


# ------------------------------------------------------------
# Seniority compatibility
# ------------------------------------------------------------

print("\nCalculating seniority compatibility...")

features["seniority_compatibility"] = features.apply(
    lambda row: seniority_compatibility(
        row.get("seniority", ""),
        row.get("experience_level", "")
    ),
    axis=1
)


# ------------------------------------------------------------
# Education compatibility
# ------------------------------------------------------------

print("\nCalculating education compatibility...")

features["education_compatibility"] = features.apply(
    lambda row: education_compatibility(
        row.get("education", ""),
        row.get("education_level", "")
    ),
    axis=1
)


# ------------------------------------------------------------
# Interaction features
# ------------------------------------------------------------

print("\nCreating interaction features...")

features["semantic_skill_interaction"] = (
    features["semantic_similarity"]
    * features["skill_coverage"]
)

features["semantic_role_interaction"] = (
    features["semantic_similarity"]
    * features["role_compatibility"]
)

features["skill_role_interaction"] = (
    features["skill_coverage"]
    * features["role_compatibility"]
)

features["experience_skill_interaction"] = (
    features["experience_compatibility"]
    * features["required_skill_match_ratio"]
)


# ------------------------------------------------------------
# Rank-normalized features
# ------------------------------------------------------------

print("\nCreating rank features...")

features["rank_score_normalized"] = (
    1 / features["rank"]
)


# ------------------------------------------------------------
# Final feature columns
# ------------------------------------------------------------

feature_columns = [
    "job_id",
    "resume_id",

    # Existing baseline signals
    "semantic_similarity",
    "skill_coverage",
    "baseline_score",

    # Skill intelligence
    "required_skill_count",
    "matched_required_skill_count",
    "missing_required_skill_count",
    "required_skill_match_ratio",

    "preferred_skill_count",
    "matched_preferred_skill_count",
    "preferred_skill_match_ratio",

    "resume_skill_count",
    "skill_union_count",
    "skill_intersection_count",

    # Candidate-job compatibility
    "role_compatibility",
    "experience_compatibility",
    "seniority_compatibility",
    "education_compatibility",

    # Interactions
    "semantic_skill_interaction",
    "semantic_role_interaction",
    "skill_role_interaction",
    "experience_skill_interaction",

    # Ranking position
    "rank_score_normalized",
]


# Keep only columns that exist
feature_columns = [
    col for col in feature_columns
    if col in features.columns
]


# Remove duplicate column names
feature_columns = list(dict.fromkeys(feature_columns))


# Select final features
final_features = features.loc[:, feature_columns].copy()


# Safety check for duplicate columns
duplicate_columns = final_features.columns[
    final_features.columns.duplicated()
].tolist()

if duplicate_columns:
    print(
        f"WARNING: Duplicate columns detected: "
        f"{duplicate_columns}"
    )

    final_features = final_features.loc[
        :, ~final_features.columns.duplicated()
    ]
# ------------------------------------------------------------
# Handle missing values
# ------------------------------------------------------------
numeric_columns = final_features.select_dtypes(
    include=[np.number]
).columns.tolist()

for column in numeric_columns:
    final_features[column] = (
        pd.to_numeric(final_features[column], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

print("\nValidating feature dataset...")

print(
    f"Final rows    : {len(final_features):,}"
)

print(
    f"Final columns : {len(final_features.columns)}"
)

print(
    f"Jobs          : {final_features['job_id'].nunique():,}"
)

print(
    f"Resumes       : {final_features['resume_id'].nunique():,}"
)

print(
    f"Missing values: "
    f"{final_features.isna().sum().sum():,}"
)


# ------------------------------------------------------------
# Feature statistics
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE SUMMARY")
print("=" * 70)

summary_columns = [
    "semantic_similarity",
    "skill_coverage",
    "required_skill_match_ratio",
    "preferred_skill_match_ratio",
    "role_compatibility",
    "experience_compatibility",
    "seniority_compatibility",
    "education_compatibility",
]

print(
    final_features[summary_columns].describe().round(4)
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

print("\nSaving feature dataset...")

final_features.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"\nSaved to:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 70)