"""Shared production ranking feature definitions for training and inference."""

import math
import re


MODEL_FEATURE_NAMES = (
    "semantic_similarity",
    "skill_coverage",
    "baseline_score",
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
    "role_compatibility",
    "experience_compatibility",
    "seniority_compatibility",
    "education_compatibility",
    "semantic_skill_interaction",
    "semantic_role_interaction",
    "skill_role_interaction",
    "experience_skill_interaction",
)


def normalize_text(value) -> str:
    if value is None:
        return ""
    try:
        if bool(value != value):
            return ""
    except (TypeError, ValueError):
        if str(value) in {"<NA>", "NaT"}:
            return ""
    value = str(value).lower()
    value = re.sub(r"[^a-z0-9+#.\-/ ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def validate_ranking_labels(dataframe, dataset_name: str) -> None:
    """Require meaningful relevance variation within at least one job group."""
    required = {"job_id", "relevance_label"}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"{dataset_name} is missing columns: {sorted(missing)}")
    if dataframe.empty:
        raise ValueError(f"{dataset_name} is empty")
    job_ids = dataframe["job_id"]
    if job_ids.isna().any() or job_ids.astype(str).str.strip().eq("").any():
        raise ValueError(f"{dataset_name} contains missing job IDs")
    labels = dataframe["relevance_label"]
    if labels.isna().any():
        raise ValueError(f"{dataset_name} contains missing relevance labels")
    try:
        numeric_labels = labels.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{dataset_name} relevance labels must be numeric") from exc
    if not numeric_labels.map(math.isfinite).all():
        raise ValueError(f"{dataset_name} contains non-finite relevance labels")
    if (numeric_labels < 0).any() or not numeric_labels.mod(1).eq(0).all():
        raise ValueError(
            f"{dataset_name} relevance labels must be non-negative integers"
        )
    variable_groups = dataframe.groupby("job_id", dropna=False)["relevance_label"].nunique()
    if not variable_groups.gt(1).any():
        raise ValueError(
            f"{dataset_name} has no job groups with varying relevance labels"
        )


def _normalize_role(value) -> str:
    text = normalize_text(value)
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


def role_compatibility(resume_role, job_title) -> float:
    """Match the role-overlap rules used to train the production ranker."""
    resume_role = _normalize_role(resume_role)
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


def experience_compatibility(resume_years, required_years) -> float:
    try:
        resume_years = float(resume_years)
        required_years = float(required_years)
    except (TypeError, ValueError):
        return 0.5
    if not math.isfinite(resume_years) or not math.isfinite(required_years):
        return 0.5
    if required_years <= 0 or resume_years >= required_years:
        return 1.0
    ratio = resume_years / required_years
    if ratio >= 0.75:
        return 0.75
    if ratio >= 0.50:
        return 0.50
    return 0.25


def seniority_compatibility(resume_seniority, job_experience_level) -> float:
    resume_level = normalize_text(resume_seniority)
    job_level = normalize_text(job_experience_level)
    if not resume_level or not job_level:
        return 0.5
    if resume_level == job_level:
        return 1.0
    compatible_pairs = {
        ("entry", "junior"),
        ("junior", "entry"),
        ("mid", "mid-level"),
        ("mid-level", "mid"),
        ("senior", "lead"),
        ("lead", "senior"),
    }
    return 0.75 if (resume_level, job_level) in compatible_pairs else 0.25


def education_compatibility(resume_education, job_education) -> float:
    resume_text = normalize_text(resume_education)
    job_text = normalize_text(job_education)
    if not resume_text or not job_text:
        return 0.5
    if job_text in resume_text or resume_text in job_text:
        return 1.0
    degree_keywords = (
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
    )
    resume_keywords = {word for word in degree_keywords if word in resume_text}
    job_keywords = {word for word in degree_keywords if word in job_text}
    return 0.75 if resume_keywords & job_keywords else 0.25


def build_ranking_feature_components(
    *,
    semantic_similarity: float,
    resume_skills,
    required_skills,
    preferred_skills,
    resume_role,
    job_title,
    resume_years,
    required_years,
    resume_seniority,
    job_seniority,
    resume_education,
    job_education,
) -> dict:
    """Build the model vector and human-readable comparisons in one place."""
    resume_skill_set = {
        normalize_text(skill) for skill in (resume_skills or []) if normalize_text(skill)
    }
    required_skill_set = {
        normalize_text(skill) for skill in (required_skills or []) if normalize_text(skill)
    }
    preferred_skill_set = {
        normalize_text(skill) for skill in (preferred_skills or []) if normalize_text(skill)
    }
    matched_required = resume_skill_set & required_skill_set
    missing_required = required_skill_set - resume_skill_set
    matched_preferred = resume_skill_set & preferred_skill_set
    missing_preferred = preferred_skill_set - resume_skill_set
    required_count = len(required_skill_set)
    preferred_count = len(preferred_skill_set)
    skill_coverage = len(matched_required) / required_count if required_count else 0.0
    preferred_coverage = len(matched_preferred) / preferred_count if preferred_count else 0.0
    role_score = role_compatibility(resume_role, job_title)
    experience_score = experience_compatibility(resume_years, required_years)
    seniority_score = seniority_compatibility(resume_seniority, job_seniority)
    education_score = education_compatibility(resume_education, job_education)
    baseline_score = 0.7 * float(semantic_similarity) + 0.3 * skill_coverage

    features = {
        "semantic_similarity": float(semantic_similarity),
        "skill_coverage": skill_coverage,
        "baseline_score": baseline_score,
        "required_skill_count": required_count,
        "matched_required_skill_count": len(matched_required),
        "missing_required_skill_count": len(missing_required),
        "required_skill_match_ratio": skill_coverage,
        "preferred_skill_count": preferred_count,
        "matched_preferred_skill_count": len(matched_preferred),
        "preferred_skill_match_ratio": preferred_coverage,
        "resume_skill_count": len(resume_skill_set),
        "skill_union_count": len(resume_skill_set | required_skill_set),
        "skill_intersection_count": len(matched_required),
        "role_compatibility": role_score,
        "experience_compatibility": experience_score,
        "seniority_compatibility": seniority_score,
        "education_compatibility": education_score,
        "semantic_skill_interaction": float(semantic_similarity) * skill_coverage,
        "semantic_role_interaction": float(semantic_similarity) * role_score,
        "skill_role_interaction": skill_coverage * role_score,
        "experience_skill_interaction": experience_score * skill_coverage,
    }
    if set(features) != set(MODEL_FEATURE_NAMES):
        raise RuntimeError("Ranking feature contract does not match the production schema")

    return {
        "features": features,
        "matched_skills": sorted(matched_required),
        "missing_skills": sorted(missing_required),
        "matched_preferred_skills": sorted(matched_preferred),
        "missing_preferred_skills": sorted(missing_preferred),
        "semantic_similarity": float(semantic_similarity),
        "skill_coverage": skill_coverage,
        "preferred_skill_coverage": preferred_coverage,
        "baseline_score": baseline_score,
    }
