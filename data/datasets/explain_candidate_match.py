import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

VALIDATION_FILE = Path(
    "data/processed/ranking_validation.csv"
)

MODEL_FILE = Path(
    "models/xgboost_ranker.json"
)

FEATURE_FILE = Path(
    "models/ranking_features.txt"
)

JOB_SKILLS_FILE = Path(
    "data/processed/job_skill_profiles.csv"
)

RESUME_SKILLS_FILE = Path(
    "data/processed/resume_skill_profiles.csv"
)

OUTPUT_FILE = Path(
    "data/processed/sample_candidate_explanation.txt"
)

# ============================================================
# LOAD DATA
# ============================================================

print("Loading validation data...")

df = pd.read_csv(VALIDATION_FILE)

job_skills_df = pd.read_csv(
    JOB_SKILLS_FILE
)

resume_skills_df = pd.read_csv(
    RESUME_SKILLS_FILE
)

# ============================================================
# LOAD MODEL
# ============================================================

model = xgb.Booster()
model.load_model(MODEL_FILE)

feature_columns = FEATURE_FILE.read_text(
    encoding="utf-8"
).splitlines()

# ============================================================
# SELECT JOB
# ============================================================

job_id = df["job_id"].iloc[0]

job_df = df[
    df["job_id"] == job_id
].copy()

print(f"Selected job: {job_id}")
print(f"Candidates: {len(job_df)}")

# ============================================================
# PREPARE FEATURES
# ============================================================

X = job_df[
    feature_columns
].apply(
    pd.to_numeric,
    errors="coerce"
)

if X.isna().sum().sum() > 0:
    raise ValueError(
        "Missing values found in candidate features."
    )

# ============================================================
# PREDICT RANKING SCORES
# ============================================================

dmatrix = xgb.DMatrix(X)

job_df["xgb_score"] = model.predict(
    dmatrix
)

job_df = job_df.sort_values(
    "xgb_score",
    ascending=False
)

job_df["xgb_rank"] = (
    np.arange(len(job_df)) + 1
)

# ============================================================
# SELECT TOP CANDIDATE
# ============================================================

top_candidate = job_df.iloc[0]

candidate_resume_id = top_candidate["resume_id"]

# IMPORTANT:
# Preserve the original index so that the candidate
# and feature row remain correctly aligned.

candidate_original_index = top_candidate.name

candidate_features = X.loc[
    [candidate_original_index]
]

print(
    f"Explaining candidate: {candidate_resume_id}"
)

# ============================================================
# LOAD SKILL PROFILES
# ============================================================

job_profile = job_skills_df[
    job_skills_df["job_id"] == job_id
]

resume_profile = resume_skills_df[
    resume_skills_df["resume_id"] == candidate_resume_id
]

if job_profile.empty:
    raise ValueError(
        f"No skill profile found for job {job_id}"
    )

if resume_profile.empty:
    raise ValueError(
        f"No skill profile found for resume "
        f"{candidate_resume_id}"
    )

# ============================================================
# PARSE SKILLS
# ============================================================

def parse_skill_string(value):
    if pd.isna(value):
        return set()

    return set(
        skill.strip().lower()
        for skill in str(value).split("|")
        if skill.strip()
    )


required_skills = parse_skill_string(
    job_profile.iloc[0]["required_skills"]
)

candidate_skills = parse_skill_string(
    resume_profile.iloc[0]["candidate_skills"]
)
# ============================================================
# MATCHED / MISSING SKILLS
# ============================================================

matched_skills = sorted(
    required_skills.intersection(
        candidate_skills
    )
)

missing_skills = sorted(
    required_skills.difference(
        candidate_skills
    )
)

skill_match_ratio = (
    len(matched_skills) / len(required_skills)
    if required_skills
    else 0
)

# ============================================================
# SHAP EXPLANATION
# ============================================================

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(
    candidate_features
)

if isinstance(shap_values, list):
    shap_values = shap_values[0]

shap_values = np.asarray(
    shap_values
).flatten()

explanation_df = pd.DataFrame({
    "feature": feature_columns,
    "feature_value": candidate_features.iloc[0].values,
    "shap_value": shap_values
})

positive = explanation_df[
    explanation_df["shap_value"] > 0
].sort_values(
    "shap_value",
    ascending=False
)

negative = explanation_df[
    explanation_df["shap_value"] < 0
].sort_values(
    "shap_value",
    ascending=True
)

# ============================================================
# BUILD REPORT
# ============================================================

report = f"""
INDIVIDUAL CANDIDATE EXPLANATION
================================

Job ID:
{job_id}

Candidate Resume ID:
{candidate_resume_id}

XGBoost Ranking:
#{int(top_candidate['xgb_rank'])}

XGBoost Score:
{top_candidate['xgb_score']:.6f}

Baseline Score:
{top_candidate['baseline_score']:.6f}

Semantic Similarity:
{top_candidate['semantic_similarity']:.6f}

Skill Coverage:
{top_candidate['skill_coverage']:.6f}

Relevance Label:
{int(top_candidate['relevance_label'])}


SKILL MATCH SUMMARY
===================

Required Skills:
{len(required_skills)}

Candidate Skills:
{len(candidate_skills)}

Matched Required Skills:
{len(matched_skills)}

Missing Required Skills:
{len(missing_skills)}

Required Skill Match Ratio:
{skill_match_ratio:.2%}


MATCHED SKILLS
==============

"""

for skill in matched_skills:
    report += f"✓ {skill}\n"

report += """

MISSING REQUIRED SKILLS
=======================

"""

for skill in missing_skills:
    report += f"✗ {skill}\n"

report += """

TOP POSITIVE FACTORS
====================

These features pushed the model prediction upward:

"""

for _, row in positive.head(8).iterrows():
    report += (
        f"+ {row['feature']}: "
        f"value={row['feature_value']:.4f}, "
        f"SHAP={row['shap_value']:.6f}\n"
    )

report += """

TOP NEGATIVE FACTORS
====================

These features pushed the model prediction downward:

"""

for _, row in negative.head(8).iterrows():
    report += (
        f"- {row['feature']}: "
        f"value={row['feature_value']:.4f}, "
        f"SHAP={row['shap_value']:.6f}\n"
    )

# ============================================================
# SAVE REPORT
# ============================================================

OUTPUT_FILE.write_text(
    report,
    encoding="utf-8"
)

print(report)

print(
    f"\nExplanation saved to:\n{OUTPUT_FILE}"
)

# ============================================================
# HUMAN-READABLE RECRUITER EXPLANATION
# ============================================================

# Determine qualitative match level.
# This is based on multiple matching signals, not the
# raw XGBoost ranking score.

if skill_match_ratio >= 0.75 and top_candidate["semantic_similarity"] >= 0.65:
    match_quality = "Strong"
elif skill_match_ratio >= 0.50 and top_candidate["semantic_similarity"] >= 0.50:
    match_quality = "Moderate"
elif skill_match_ratio >= 0.25 or top_candidate["semantic_similarity"] >= 0.45:
    match_quality = "Limited"
else:
    match_quality = "Low"

# Build explanation
human_report = f"""
RECRUITER-FRIENDLY CANDIDATE EXPLANATION
========================================

Candidate:
{candidate_resume_id}

Job:
{job_id}

Ranking:
#{int(top_candidate['xgb_rank'])}

Match Quality:
{match_quality}


MATCH OVERVIEW
==============

Required skills: {len(required_skills)}
Candidate skills: {len(candidate_skills)}
Required skills matched: {len(matched_skills)}
Required skills missing: {len(missing_skills)}

Skill coverage:
{skill_match_ratio:.1%}

Semantic similarity:
{top_candidate['semantic_similarity']:.3f}

Seniority compatibility:
{top_candidate['seniority_compatibility']:.3f}


MATCHED SKILLS
==============

"""

for skill in matched_skills:
    human_report += f"✓ {skill}\n"

human_report += """

MISSING REQUIRED SKILLS
=======================

"""

for skill in missing_skills:
    human_report += f"✗ {skill}\n"

human_report += """

KEY POSITIVE MODEL FACTORS
==========================

"""

for _, row in positive.head(5).iterrows():
    human_report += (
        f"↑ {row['feature']} "
        f"(SHAP: {row['shap_value']:.3f})\n"
    )

human_report += """

KEY NEGATIVE MODEL FACTORS
==========================

"""

for _, row in negative.head(5).iterrows():
    human_report += (
        f"↓ {row['feature']} "
        f"(SHAP: {row['shap_value']:.3f})\n"
    )

human_report += f"""

RECRUITER SUMMARY
=================

The candidate ranks #{int(top_candidate['xgb_rank'])}
within the evaluated candidate pool.

The candidate shows a
{top_candidate['seniority_compatibility']:.0%}
seniority compatibility and
{top_candidate['semantic_similarity']:.0%}
semantic similarity.

However, only {len(matched_skills)} of
{len(required_skills)} required skills are currently matched.

The ranking model therefore considers this candidate
relatively stronger than the other candidates in this pool,
but the limited direct skill and role alignment should be
reviewed before making a hiring decision.

IMPORTANT:
This explanation is model-assisted and should support
human recruiter review. It is not an automated hiring
decision.
"""

print(human_report)

# Save human-readable explanation separately
HUMAN_OUTPUT_FILE = Path(
    "data/processed/recruiter_candidate_explanation.txt"
)

HUMAN_OUTPUT_FILE.write_text(
    human_report,
    encoding="utf-8"
)

print(
    f"\nRecruiter explanation saved to:\n"
    f"{HUMAN_OUTPUT_FILE}"
)

print("\nSTEP 6.9.3 COMPLETE")