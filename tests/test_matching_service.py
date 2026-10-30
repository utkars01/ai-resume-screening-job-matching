import numpy as np
import pandas as pd
import pytest

from backend.app.services.matching_service import cosine_similarity
from backend.app.services.ranking_feature_contract import (
    MODEL_FEATURE_NAMES,
    build_ranking_feature_components,
    education_compatibility,
    experience_compatibility,
    normalize_text,
    role_compatibility,
    seniority_compatibility,
    validate_ranking_labels,
)


def test_normalize_text_handles_missing_values():
    assert normalize_text("  Python   DEVELOPER! ") == "python developer"
    assert normalize_text(None) == ""
    assert normalize_text(float("nan")) == ""
    assert normalize_text(pd.NA) == ""


@pytest.mark.parametrize(
    ("resume_role", "job_title", "expected"),
    [
        ("Software Engineer", "Software Engineer II", 1.0),
        ("Backend Developer", "Backend Engineer", 1.0),
        ("Data Scientist", "AI Engineer", 0.0),
        ("Data Analyst", "Data Engineer", 0.75),
        (None, "Data Engineer", 0.0),
    ],
)
def test_role_compatibility(resume_role, job_title, expected):
    assert role_compatibility(resume_role, job_title) == expected


@pytest.mark.parametrize(
    ("candidate_years", "required_years", "expected"),
    [(0, 0, 1.0), (3, 2, 1.0), (3, 4, 0.75), (2, 4, 0.5), (1, 4, 0.25), (None, 2, 0.5)],
)
def test_experience_compatibility(candidate_years, required_years, expected):
    from backend.app.services.ranking_feature_contract import experience_compatibility

    assert experience_compatibility(candidate_years, required_years) == expected


def test_seniority_compatibility_matches_training_rules():
    assert seniority_compatibility("senior", "lead") == 0.75
    assert seniority_compatibility("entry", "intern") == 0.25
    assert seniority_compatibility(None, "intern") == 0.5


def test_education_compatibility_matches_training_rules():
    assert education_compatibility("B.Tech", "B.Tech") == 1.0
    assert education_compatibility("Bachelor of Science in Computer Science", "Computer Science degree") == 0.75
    assert education_compatibility("High School", "PhD") == 0.25
    assert education_compatibility(None, "B.Tech") == 0.5


def test_feature_contract_is_ordered_and_uses_shared_formulas():
    components = build_ranking_feature_components(
        semantic_similarity=0.8,
        resume_skills=["Python", "SQL"],
        required_skills=["Python", "Java"],
        preferred_skills=["SQL"],
        resume_role="Data Engineer",
        job_title="Data Engineer",
        resume_years=3,
        required_years=4,
        resume_seniority="mid",
        job_seniority="mid-level",
        resume_education="Bachelor of Science",
        job_education="Bachelor",
    )
    features = components["features"]
    assert tuple(features) == MODEL_FEATURE_NAMES
    assert features["role_compatibility"] == 1.0
    assert features["experience_compatibility"] == 0.75
    assert features["skill_coverage"] == 0.5
    assert features["baseline_score"] == pytest.approx(0.71)
    assert features["experience_skill_interaction"] == 0.375


def test_validate_ranking_labels_rejects_single_label_groups():
    single_label = pd.DataFrame({"job_id": ["a", "a"], "relevance_label": [0, 0]})
    with pytest.raises(ValueError, match="no job groups with varying"):
        validate_ranking_labels(single_label, "validation split")


def test_validate_ranking_labels_accepts_variation_within_jobs():
    data = pd.DataFrame({"job_id": ["a", "a"], "relevance_label": [0, 2]})
    validate_ranking_labels(data, "training split")


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (
            pd.DataFrame({"job_id": ["a", None], "relevance_label": [0, 2]}),
            "missing job IDs",
        ),
        (
            pd.DataFrame({"job_id": ["a", "a"], "relevance_label": [-1, 2]}),
            "non-negative integers",
        ),
        (
            pd.DataFrame({"job_id": ["a", "a"], "relevance_label": [0, 1.5]}),
            "non-negative integers",
        ),
    ],
)
def test_validate_ranking_labels_rejects_invalid_group_data(data, message):
    with pytest.raises(ValueError, match=message):
        validate_ranking_labels(data, "ranking split")


def test_cosine_similarity_identical_vectors():
    assert cosine_similarity([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector():
    assert cosine_similarity([0, 0], [1, 0]) == 0.0


def test_cosine_similarity_opposite_vectors():
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)


def test_cosine_similarity_rejects_empty_vectors():
    with pytest.raises(ValueError, match="empty embeddings"):
        cosine_similarity([], [1])


def test_cosine_similarity_rejects_dimension_mismatch():
    with pytest.raises(ValueError, match="different dimensions"):
        cosine_similarity([1, 2], [1, 2, 3])


def test_cosine_similarity_rejects_non_finite_values():
    with pytest.raises(ValueError, match="non-finite"):
        cosine_similarity([1, np.nan], [1, 2])
