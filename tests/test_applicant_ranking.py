from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from backend.app.auth import get_current_recruiter
from backend.app.api import applicants as applicants_api
from backend.app.main import app
from backend.app.models.applicant import Applicant
from backend.app.models.applicant_ai_profile import ApplicantAIProfile
from backend.app.models.job import Job
from backend.app.models.job_ai_profile import JobAIProfile


def test_applicant_ranking_returns_descending_raw_model_scores(monkeypatch):
    job = SimpleNamespace(id=12, title="Data Engineer", status="active")
    job_profile = SimpleNamespace(job_id=12)
    applicants = [
        SimpleNamespace(
            id=21,
            job_id=12,
            original_filename="candidate-a.pdf",
            role="Data Engineer",
            years_experience=4,
            seniority="mid",
            education="B.S. Computer Science",
            extracted_text="resume a",
        ),
        SimpleNamespace(
            id=22,
            job_id=12,
            original_filename="candidate-b.pdf",
            role=None,
            years_experience=None,
            seniority=None,
            education=None,
            extracted_text="resume b",
        ),
    ]
    profiles = {
        21: SimpleNamespace(applicant_id=21),
        22: SimpleNamespace(applicant_id=22),
    }
    db = MagicMock()
    schema_result = MagicMock()
    schema_result.scalars.return_value.all.return_value = [
        "screening_runs",
        "screening_results",
        "applicant_shortlists",
    ]
    run_result = MagicMock()
    run_result.scalar_one.return_value = 345
    db.execute.side_effect = [schema_result, run_result, MagicMock(), MagicMock()]

    def query(model):
        result = MagicMock()
        if model is Job:
            result.filter.return_value.first.return_value = job
        elif model is JobAIProfile:
            result.filter.return_value.first.return_value = job_profile
        elif model is Applicant:
            result.filter.return_value.order_by.return_value.all.return_value = applicants
        elif model is ApplicantAIProfile:
            result.filter.return_value.first.side_effect = [profiles[21], profiles[22]]
        return result

    db.query.side_effect = query

    def calculate_features(candidate, **_):
        score = 0.91 if candidate.id == 21 else 0.31
        return {
            "features": {
                "role_compatibility": score,
                "experience_compatibility": score,
                "seniority_compatibility": score,
                "education_compatibility": score,
            },
            "semantic_similarity": score,
            "skill_coverage": score,
            "baseline_score": score,
            "preferred_skill_coverage": 0.0,
            "matched_skills": ["python"] if candidate.id == 21 else [],
            "missing_skills": ["spark"],
            "matched_preferred_skills": [],
            "missing_preferred_skills": [],
        }

    def override_get_db():
        yield db

    app.dependency_overrides[applicants_api.get_db] = override_get_db
    app.dependency_overrides[get_current_recruiter] = lambda: SimpleNamespace(
        id=10, role="recruiter"
    )
    monkeypatch.setattr(
        applicants_api.matching_service, "calculate_features", calculate_features
    )
    monkeypatch.setattr(
        applicants_api.matching_service,
        "predict_score",
        lambda features: features["role_compatibility"],
    )

    try:
        response = TestClient(app).post("/api/applicants/jobs/12/rank")
    finally:
        app.dependency_overrides.pop(applicants_api.get_db, None)
        app.dependency_overrides.pop(get_current_recruiter, None)

    assert response.status_code == 200
    body = response.json()
    assert body["ranking_score_type"] == "raw_xgboost_score_not_probability"
    assert body["screening_run_id"] == 345
    assert [row["applicant_id"] for row in body["results"]] == [21, 22]
    assert [row["rank"] for row in body["results"]] == [1, 2]
    assert body["results"][0]["role"] == "Data Engineer"
    assert body["results"][0]["matched_skills"] == ["python"]
    assert body["results"][0]["missing_skills"] == ["spark"]
    assert body["results"][0]["matched_preferred_skills"] == []
    assert body["results"][0]["missing_preferred_skills"] == []
    assert body["results"][0]["role_compatibility"] == 0.91
    assert body["results"][0]["experience_compatibility"] == 0.91
    assert body["results"][1]["role"] is None
    assert body["results"][1]["years_experience"] is None
    assert body["results"][1]["education"] is None
