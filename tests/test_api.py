from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_health_and_root_remain_public():
    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 200


def test_job_and_applicant_routes_require_recruiter_authentication():
    assert client.get("/api/jobs/").status_code == 401
    assert client.post("/api/applicants/jobs/12/rank").status_code == 401
    assert client.get("/api/auth/me").status_code == 401


def test_candidate_account_routes_are_not_exposed():
    paths = (
        "/api/users/",
        "/api/candidates/",
        "/api/resumes/",
        "/api/matching/jobs/12/rank-candidates",
        "/api/database/test",
    )
    for path in paths:
        assert client.get(path).status_code == 404
