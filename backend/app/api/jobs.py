from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session
from typing import Literal

from backend.app.database.session import get_db
from backend.app.auth import get_current_recruiter
from backend.app.models.job import Job
from backend.app.models.job_ai_profile import JobAIProfile
from backend.app.models.user import User
from backend.app.schemas.job import JobCreate, JobResponse
from backend.app.schemas.job_ai_profile import JobAIProfileResponse
from backend.app.services.job_processor import process_job


router = APIRouter(
    prefix="/api/jobs",
    tags=["Jobs"],
)


class JobUpdate(BaseModel):
    title: str | None = None
    company_name: str | None = None
    location: str | None = None
    industry: str | None = None
    description: str | None = None
    required_skills: str | None = None
    preferred_skills: str | None = None
    experience_required: int | None = Field(default=None, ge=0)
    seniority_required: str | None = None
    education_required: str | None = None
    status: Literal["active", "closed"] | None = None

    model_config = ConfigDict(extra="forbid")


@router.post(
    "/",
    response_model=JobResponse,
    status_code=201,
)
def create_job(
    job_data: JobCreate,
    db: Session = Depends(get_db),
    recruiter: User = Depends(get_current_recruiter),
):
    # The owner comes from the authenticated session, never request data.
    job = Job(
        recruiter_id=recruiter.id,
        title=job_data.title,
        company_name=job_data.company_name,
        location=job_data.location,
        industry=job_data.industry,
        description=job_data.description,
        required_skills=job_data.required_skills,
        preferred_skills=job_data.preferred_skills,
        experience_required=job_data.experience_required,
        seniority_required=job_data.seniority_required,
        education_required=job_data.education_required,
    )

    db.add(job)

    try:
        # Process before committing so a slow or failed embedding cannot leave
        # a visible job without its AI profile. The processor flushes to obtain
        # the job ID and commits the job and profile together on success.
        process_job(job=job, db=db)

        return job

    except Exception:
        db.rollback()

        # Clean up the newly created job if AI processing fails.
        if job.id is not None:
            existing_job = (
                db.query(Job)
                .filter(Job.id == job.id)
                .first()
            )

            if existing_job is not None:
                db.delete(existing_job)
                db.commit()

        raise HTTPException(
            status_code=500,
            detail="Job creation or AI processing failed.",
        )


@router.get(
    "/",
    response_model=list[JobResponse],
)
def get_jobs(
    db: Session = Depends(get_db),
    recruiter: User = Depends(get_current_recruiter),
):
    return db.query(Job).filter(Job.recruiter_id == recruiter.id).all()


@router.get(
    "/{job_id}",
    response_model=JobResponse,
)
def get_job(
    job_id: int,
    db: Session = Depends(get_db),
    recruiter: User = Depends(get_current_recruiter),
):
    job = (
        db.query(Job)
        .filter(Job.id == job_id, Job.recruiter_id == recruiter.id)
        .first()
    )

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found",
        )

    return job


@router.patch(
    "/{job_id}",
    response_model=JobResponse,
)
def update_job(
    job_id: int,
    job_data: JobUpdate,
    db: Session = Depends(get_db),
    recruiter: User = Depends(get_current_recruiter),
):
    # 1. Find the existing job.
    job = (
        db.query(Job)
        .filter(Job.id == job_id, Job.recruiter_id == recruiter.id)
        .first()
    )

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found",
        )

    # 2. Apply only fields supplied by the caller.
    updates = job_data.model_dump(exclude_unset=True)

    for field, value in updates.items():
        setattr(job, field, value)

    matching_fields = {
        "title",
        "description",
        "required_skills",
        "preferred_skills",
        "experience_required",
        "seniority_required",
        "education_required",
    }

    try:
        if matching_fields.intersection(updates):
            # Commit matching inputs and the refreshed AI profile together.
            process_job(job=job, db=db)
        else:
            db.commit()
            db.refresh(job)

        return job

    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Job update or AI reprocessing failed.",
        )


@router.get(
    "/{job_id}/ai-profile",
    response_model=JobAIProfileResponse,
)
def get_job_ai_profile(
    job_id: int,
    db: Session = Depends(get_db),
    recruiter: User = Depends(get_current_recruiter),
):
    # 1. Verify that the job exists.
    job = (
        db.query(Job)
        .filter(Job.id == job_id, Job.recruiter_id == recruiter.id)
        .first()
    )

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found",
        )

    # 2. Retrieve the AI profile.
    ai_profile = (
        db.query(JobAIProfile)
        .filter(JobAIProfile.job_id == job_id)
        .first()
    )

    if ai_profile is None:
        raise HTTPException(
            status_code=404,
            detail="AI profile not found for this job",
        )

    return ai_profile


@router.post(
    "/{job_id}/reprocess",
    response_model=JobAIProfileResponse,
)
def reprocess_job(
    job_id: int,
    db: Session = Depends(get_db),
    recruiter: User = Depends(get_current_recruiter),
):
    # 1. Verify that the job exists.
    job = (
        db.query(Job)
        .filter(Job.id == job_id, Job.recruiter_id == recruiter.id)
        .first()
    )

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found",
        )

    # 2. Reprocess the existing job without creating a duplicate.
    try:
        process_job(job=job, db=db)
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Job reprocessing failed.",
        )

    # 3. Return the refreshed profile.
    ai_profile = (
        db.query(JobAIProfile)
        .filter(JobAIProfile.job_id == job_id)
        .first()
    )

    if ai_profile is None:
        raise HTTPException(
            status_code=500,
            detail="AI profile was not created.",
        )

    return ai_profile
