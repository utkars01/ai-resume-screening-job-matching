from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest
from io import BytesIO
from zipfile import ZipFile

from fastapi.testclient import TestClient

from backend.app.api import applicants as applicants_api
from backend.app.api import jobs as jobs_api
from backend.app.auth import get_current_recruiter
from backend.app.main import app
from backend.app.models.job import Job


def test_job_lookup_filters_by_authenticated_recruiter():
    db = MagicMock()
    query = db.query.return_value
    query.filter.return_value.first.return_value = None

    def override_db():
        yield db

    app.dependency_overrides[jobs_api.get_db] = override_db
    app.dependency_overrides[get_current_recruiter] = lambda: SimpleNamespace(
        id=73, role="recruiter"
    )
    try:
        response = TestClient(app).get("/api/jobs/91")
    finally:
        app.dependency_overrides.pop(jobs_api.get_db, None)
        app.dependency_overrides.pop(get_current_recruiter, None)

    assert response.status_code == 404
    filters = query.filter.call_args.args
    assert len(filters) == 2
    owner_filter = filters[1].compile()
    assert "jobs.recruiter_id" in str(owner_filter)
    assert 73 in owner_filter.params.values()


def test_bulk_upload_hides_jobs_owned_by_another_recruiter(tmp_path, monkeypatch):
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None

    def override_db():
        yield db

    app.dependency_overrides[applicants_api.get_db] = override_db
    app.dependency_overrides[get_current_recruiter] = lambda: SimpleNamespace(
        id=73, role="recruiter"
    )
    monkeypatch.setattr(applicants_api, "UPLOAD_DIR", tmp_path)
    try:
        buffer = BytesIO()
        with ZipFile(buffer, "w") as archive:
            archive.writestr("[Content_Types].xml", "<Types/>")
            archive.writestr("word/document.xml", "<document/>")
        response = TestClient(app).post(
            "/api/applicants/bulk-upload?job_id=91",
            files={"files": ("resume.docx", buffer.getvalue(), "application/octet-stream")},
        )
    finally:
        app.dependency_overrides.pop(applicants_api.get_db, None)
        app.dependency_overrides.pop(get_current_recruiter, None)

    assert response.status_code == 404
    filters = db.query.return_value.filter.call_args.args
    compiled = [item.compile() for item in filters]
    assert any("jobs.recruiter_id" in str(item) for item in compiled)
    assert any(73 in item.params.values() for item in compiled)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("method", "path", "kwargs"),
    [
        ("GET", "/api/applicants/jobs/91", {}),
        ("POST", "/api/applicants/jobs/91/rank", {}),
        ("GET", "/api/applicants/jobs/91/screening-runs", {}),
        ("GET", "/api/applicants/jobs/91/shortlist", {}),
        (
            "POST",
            "/api/applicants/jobs/91/applicants/112/shortlist",
            {"json": {}},
        ),
        (
            "DELETE",
            "/api/applicants/jobs/91/applicants/112/shortlist",
            {},
        ),
    ],
)
def test_job_scoped_applicant_routes_hide_another_recruiters_job(
    method, path, kwargs
):
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None

    def override_db():
        yield db

    app.dependency_overrides[applicants_api.get_db] = override_db
    app.dependency_overrides[get_current_recruiter] = lambda: SimpleNamespace(
        id=73, role="recruiter"
    )
    try:
        response = TestClient(app).request(method, path, **kwargs)
    finally:
        app.dependency_overrides.pop(applicants_api.get_db, None)
        app.dependency_overrides.pop(get_current_recruiter, None)

    assert response.status_code == 404
    filters = db.query.return_value.filter.call_args.args
    compiled = [clause.compile() for clause in filters]
    assert any("jobs.recruiter_id" in str(clause) for clause in compiled)
    assert any(73 in clause.params.values() for clause in compiled)
    db.commit.assert_not_called()
