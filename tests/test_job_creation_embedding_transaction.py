from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
from fastapi import HTTPException

from backend.app.api import jobs as jobs_api
from backend.app.models.job_ai_profile import JobAIProfile
from backend.app.schemas.job import JobCreate
from backend.app.services import job_processor


def _job_payload():
    return JobCreate(
        title="Synthetic Data Engineer",
        company_name="Synthetic Test Lab",
        description="Build data services with Python and SQL.",
    )


def test_create_job_does_not_commit_before_embedding_processing(monkeypatch):
    db = MagicMock()

    def process_pending_job(*, job, db):
        assert job.id is None
        db.flush()
        job.id = 87
        db.commit()

    monkeypatch.setattr(jobs_api, "process_job", process_pending_job)

    result = jobs_api.create_job(_job_payload(), db, SimpleNamespace(id=42))

    assert result.id == 87
    db.add.assert_called_once_with(result)
    db.commit.assert_called_once()
    db.rollback.assert_not_called()


def test_create_job_leaves_no_committed_record_when_embedding_processing_fails(
    monkeypatch,
):
    db = MagicMock()

    def fail_processing(**_kwargs):
        raise RuntimeError("synthetic embedding failure")

    monkeypatch.setattr(jobs_api, "process_job", fail_processing)

    with pytest.raises(HTTPException) as error:
        jobs_api.create_job(_job_payload(), db, SimpleNamespace(id=42))

    assert error.value.status_code == 500
    db.add.assert_called_once()
    db.commit.assert_not_called()
    db.rollback.assert_called_once()
    db.query.assert_not_called()


def test_matching_job_update_does_not_commit_before_reprocessing(monkeypatch):
    job = SimpleNamespace(
        id=87,
        recruiter_id=42,
        title="Old title",
        description="Existing description",
        required_skills="Python",
        preferred_skills=None,
        experience_required=None,
        seniority_required=None,
        education_required=None,
        status="active",
    )
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = job

    def reprocess_before_commit(*, job, db):
        assert job.title == "New title"
        db.commit()

    monkeypatch.setattr(jobs_api, "process_job", reprocess_before_commit)
    result = jobs_api.update_job(
        87,
        jobs_api.JobUpdate(title="New title"),
        db,
        SimpleNamespace(id=42),
    )

    assert result.title == "New title"
    db.commit.assert_called_once()
    db.refresh.assert_not_called()


def test_new_job_is_flushed_after_embedding_and_before_profile_lookup(monkeypatch):
    job = SimpleNamespace(
        id=None,
        title="Synthetic Data Engineer",
        description="Python and SQL data systems.",
        required_skills="Python, SQL",
        preferred_skills=None,
        experience_required=None,
        seniority_required=None,
        education_required=None,
    )
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None
    db.flush.side_effect = lambda: setattr(job, "id", 87)
    events = Mock()
    events.attach_mock(db.flush, "flush")
    events.attach_mock(db.query, "query")
    events.attach_mock(db.commit, "commit")

    monkeypatch.setattr(
        job_processor.skill_service,
        "extract_skills",
        lambda _text: ["python", "sql"],
    )
    monkeypatch.setattr(
        job_processor.embedding_service,
        "generate_embedding",
        lambda _text: [0.0] * 384,
    )

    result = job_processor.process_job(job, db)

    assert result is job
    assert job.id == 87
    assert any(
        isinstance(call.args[0], JobAIProfile)
        for call in db.add.call_args_list
    )
    call_names = [call[0] for call in events.mock_calls]
    assert call_names.index("flush") < call_names.index("query")
    assert call_names.index("query") < call_names.index("commit")
