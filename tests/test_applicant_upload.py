import asyncio
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from zipfile import ZipFile

from fastapi.testclient import TestClient

from backend.app.auth import get_current_recruiter
from backend.app.api import applicants as applicants_api
from backend.app.main import app
from backend.app.models.applicant import Applicant
from backend.app.models.job import Job
from backend.app.models.job_ai_profile import JobAIProfile


def _docx_payload():
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", "<document/>")
    return buffer.getvalue()


def test_bulk_upload_persists_metadata_and_reports_duplicate(
    monkeypatch, tmp_path
):
    job = SimpleNamespace(id=7, status="active")
    job_profile = SimpleNamespace(job_id=7)
    added = []
    db = MagicMock()
    db.add.side_effect = added.append

    def query(model):
        result = MagicMock()
        if model is Job:
            result.filter.return_value.first.return_value = job
        elif model is JobAIProfile:
            result.filter.return_value.first.return_value = job_profile
        return result

    db.query.side_effect = query

    def refresh(row):
        if isinstance(row, Applicant):
            row.id = 101

    db.refresh.side_effect = refresh

    def override_get_db():
        yield db

    app.dependency_overrides[applicants_api.get_db] = override_get_db
    app.dependency_overrides[get_current_recruiter] = lambda: SimpleNamespace(
        id=10, role="recruiter"
    )
    monkeypatch.setattr(applicants_api, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(
        applicants_api,
        "extract_text",
        lambda *_: """Alex Example
Role: Senior Cloud Engineer
6+ years of experience
M.S. Computer Science
""",
    )
    monkeypatch.setattr(
        applicants_api.skill_service, "extract_skills", lambda _: ["Python"]
    )
    monkeypatch.setattr(
        applicants_api.embedding_service,
        "generate_embedding",
        lambda _: [0.0] * 384,
    )

    try:
        payload = _docx_payload()
        response = TestClient(app).post(
            "/api/applicants/bulk-upload?job_id=7",
            files=[
                ("files", ("alex.docx", payload, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
                ("files", ("alex-copy.docx", payload, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
            ],
        )
    finally:
        app.dependency_overrides.pop(applicants_api.get_db, None)
        app.dependency_overrides.pop(get_current_recruiter, None)

    assert response.status_code == 201
    body = response.json()
    assert body["completed"] == 1
    assert body["failed"] == 1
    assert body["results"][0]["role"] == "Senior Cloud Engineer"
    assert body["results"][0]["years_experience"] == 6
    assert body["results"][0]["seniority"] == "senior"
    assert body["results"][0]["education"] == "M.S. Computer Science"
    assert "duplicated" in body["results"][1]["message"]

    applicants = [row for row in added if isinstance(row, Applicant)]
    assert len(applicants) == 1
    assert applicants[0].processing_status == "completed"
    assert applicants[0].role == "Senior Cloud Engineer"
    assert applicants[0].years_experience == 6
    assert applicants[0].seniority == "senior"
    assert applicants[0].education == "M.S. Computer Science"


def test_bulk_upload_reads_only_one_byte_past_file_limit(monkeypatch, tmp_path):
    job = SimpleNamespace(id=7, status="active")
    job_profile = SimpleNamespace(job_id=7)
    db = MagicMock()

    def query(model):
        result = MagicMock()
        if model is Job:
            result.filter.return_value.first.return_value = job
        elif model is JobAIProfile:
            result.filter.return_value.first.return_value = job_profile
        return result

    db.query.side_effect = query
    upload = MagicMock()
    upload.filename = "too-large.pdf"
    upload.read = AsyncMock(return_value=b"x" * (applicants_api.MAX_FILE_SIZE + 1))
    upload.close = AsyncMock()
    monkeypatch.setattr(applicants_api, "UPLOAD_DIR", tmp_path)

    response = asyncio.run(
        applicants_api.bulk_upload_applicants(
            files=[upload],
            job_id=7,
            db=db,
            recruiter=SimpleNamespace(id=10, role="recruiter"),
        )
    )

    assert upload.read.await_args.args == (applicants_api.MAX_FILE_SIZE + 1,)
    assert response["failed"] == 1
    assert response["results"][0]["message"] == "File exceeds the 5 MB limit."
    assert list(tmp_path.iterdir()) == []


def test_bulk_upload_explains_that_scanned_pdfs_need_ocr(monkeypatch, tmp_path):
    job = SimpleNamespace(id=7, status="active")
    job_profile = SimpleNamespace(job_id=7)
    db = MagicMock()

    def query(model):
        result = MagicMock()
        if model is Job:
            result.filter.return_value.first.return_value = job
        elif model is JobAIProfile:
            result.filter.return_value.first.return_value = job_profile
        return result

    db.query.side_effect = query
    def override_get_db():
        yield db

    app.dependency_overrides[applicants_api.get_db] = override_get_db
    app.dependency_overrides[get_current_recruiter] = lambda: SimpleNamespace(
        id=10, role="recruiter"
    )
    monkeypatch.setattr(applicants_api, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(applicants_api, "extract_text", lambda *_: "")

    try:
        response = TestClient(app).post(
            "/api/applicants/bulk-upload?job_id=7",
            files=[("files", ("scanned.pdf", b"%PDF-1.4 image", "application/pdf"))],
        )
    finally:
        app.dependency_overrides.pop(applicants_api.get_db, None)
        app.dependency_overrides.pop(get_current_recruiter, None)

    assert response.status_code == 201
    result = response.json()["results"][0]
    assert result["status"] == "failed"
    assert "image-only" in result["message"]
    assert "OCR is not enabled" in result["message"]
    assert "text-based PDF or DOCX" in result["message"]
    assert list(tmp_path.iterdir()) == []
