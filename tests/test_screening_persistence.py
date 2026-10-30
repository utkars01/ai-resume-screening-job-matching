from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from backend.app.api import applicants as applicants_api
from backend.app.auth import get_current_recruiter
from backend.app.main import app
from backend.app.models.applicant import Applicant
from backend.app.models.job import Job
from backend.app.models.job_ai_profile import JobAIProfile


def _override_db(db):
    def dependency():
        yield db

    return dependency


def _set_request_overrides(db):
    app.dependency_overrides[applicants_api.get_db] = _override_db(db)
    app.dependency_overrides[get_current_recruiter] = lambda: SimpleNamespace(
        id=10, role="recruiter"
    )


def _clear_request_overrides():
    app.dependency_overrides.pop(applicants_api.get_db, None)
    app.dependency_overrides.pop(get_current_recruiter, None)


def test_ranking_fails_closed_when_persistence_schema_is_missing():
    job = SimpleNamespace(id=12, title="Data Engineer", status="active")
    job_profile = SimpleNamespace(job_id=12)
    db = MagicMock()
    missing_tables = MagicMock()
    missing_tables.scalars.return_value.all.return_value = []
    db.execute.return_value = missing_tables

    def query(model):
        query_result = MagicMock()
        if model is Job:
            query_result.filter.return_value.first.return_value = job
        elif model is JobAIProfile:
            query_result.filter.return_value.first.return_value = job_profile
        return query_result

    db.query.side_effect = query
    _set_request_overrides(db)
    try:
        response = TestClient(app).post("/api/applicants/jobs/12/rank")
    finally:
        _clear_request_overrides()

    assert response.status_code == 503
    assert "reviewed database schema update" in response.json()["detail"]
    assert db.execute.call_count == 1


def test_screening_history_fails_closed_when_history_tables_are_missing():
    job = SimpleNamespace(id=12, title="Data Engineer", status="active")
    db = MagicMock()
    missing_tables = MagicMock()
    missing_tables.scalars.return_value.all.return_value = ["applicant_shortlists"]
    db.execute.return_value = missing_tables
    db.query.return_value.filter.return_value.first.return_value = job

    _set_request_overrides(db)
    try:
        response = TestClient(app).get("/api/applicants/jobs/12/screening-runs")
    finally:
        _clear_request_overrides()

    assert response.status_code == 503
    assert "reviewed database schema update" in response.json()["detail"]
    assert db.execute.call_count == 1


def test_shortlist_add_is_job_scoped_and_persists_note():
    job = SimpleNamespace(id=12)
    applicant = SimpleNamespace(id=21, job_id=12)
    db = MagicMock()

    def query(model):
        result = MagicMock()
        if model is Job:
            result.filter.return_value.first.return_value = job
        elif model is Applicant:
            result.filter.return_value.first.return_value = applicant
        return result

    db.query.side_effect = query
    schema_result = MagicMock()
    schema_result.scalars.return_value.all.return_value = ["applicant_shortlists"]
    insert_result = MagicMock()
    insert_result.mappings.return_value.one.return_value = {
        "id": 88,
        "recruiter_id": 10,
        "job_id": 12,
        "applicant_id": 21,
        "note": "Review portfolio",
        "created_at": datetime(2026, 10, 27),
    }
    db.execute.side_effect = [schema_result, insert_result]

    _set_request_overrides(db)
    try:
        response = TestClient(app).post(
            "/api/applicants/jobs/12/applicants/21/shortlist",
            json={"note": "Review portfolio"},
        )
    finally:
        _clear_request_overrides()

    assert response.status_code == 201
    assert response.json()["applicant_id"] == 21
    assert response.json()["note"] == "Review portfolio"
    insert_parameters = db.execute.call_args_list[-1].args[1]
    assert insert_parameters == {
        "recruiter_id": 10,
        "job_id": 12,
        "applicant_id": 21,
        "note": "Review portfolio",
    }
    db.commit.assert_called_once()


def test_shortlist_add_rejects_applicant_from_another_job():
    job = SimpleNamespace(id=12)
    db = MagicMock()

    def query(model):
        result = MagicMock()
        if model is Job:
            result.filter.return_value.first.return_value = job
        elif model is Applicant:
            result.filter.return_value.first.return_value = None
        return result

    db.query.side_effect = query
    _set_request_overrides(db)
    try:
        response = TestClient(app).post(
            "/api/applicants/jobs/12/applicants/21/shortlist",
            json={},
        )
    finally:
        _clear_request_overrides()

    assert response.status_code == 404
    assert response.json()["detail"] == "Applicant not found for this job."
    db.execute.assert_not_called()
    db.commit.assert_not_called()


def test_shortlist_remove_commits_only_one_matching_entry():
    job = SimpleNamespace(id=12)
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = job
    schema_result = MagicMock()
    schema_result.scalars.return_value.all.return_value = ["applicant_shortlists"]
    delete_result = MagicMock()
    delete_result.rowcount = 1
    db.execute.side_effect = [schema_result, delete_result]

    _set_request_overrides(db)
    try:
        response = TestClient(app).delete(
            "/api/applicants/jobs/12/applicants/21/shortlist"
        )
    finally:
        _clear_request_overrides()

    assert response.status_code == 204
    delete_parameters = db.execute.call_args_list[-1].args[1]
    assert delete_parameters == {
        "recruiter_id": 10,
        "job_id": 12,
        "applicant_id": 21,
    }
    db.commit.assert_called_once()
