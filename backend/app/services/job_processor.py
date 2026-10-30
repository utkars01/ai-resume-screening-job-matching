import logging

from sqlalchemy.orm import Session
from datetime import datetime
from backend.app.models.job import Job
from backend.app.models.job_ai_profile import JobAIProfile
from backend.app.services.embedding_service import embedding_service
from backend.app.services.skill_service import skill_service


PROCESSING_VERSION = "v2"
logger = logging.getLogger(__name__)


def process_job(
    job: Job,
    db: Session,
) -> Job:
    """
    Process a job using the AI pipeline.

    Processing:
    1. Validate the job description.
    2. Extract required skills.
    3. Extract preferred skills separately.
    4. Generate a 384-dimensional job embedding.
    5. Create or update the JobAIProfile.
    """

    if not job.description or not job.description.strip():
        raise ValueError("Job description cannot be empty.")

    try:
        # 1. Extract skills from the job description.
        description_skills = skill_service.extract_skills(
            job.description
        )

        # 2. Extract explicitly supplied required skills.
        explicit_required_skills = []

        if job.required_skills:
            explicit_required_skills = (
                skill_service.extract_skills(
                    job.required_skills
                )
            )

        # Combine required skills from both sources.
        required_skills = sorted(
            set(description_skills)
            | set(explicit_required_skills)
        )

        # 3. Extract preferred skills separately.
        preferred_skills = []

        if job.preferred_skills:
            preferred_skills = sorted(
                set(
                    skill_service.extract_skills(
                        job.preferred_skills
                    )
                )
            )

        # 4. Generate the job embedding.
        embedding_parts = [
            job.title or "",
            job.description,
            job.required_skills or "",
            job.preferred_skills or "",
        ]

        embedding_text = "\n".join(
            part.strip()
            for part in embedding_parts
            if part and part.strip()
        )

        embedding = embedding_service.generate_embedding(
            embedding_text
        )

        if embedding is None or len(embedding) == 0:
            raise ValueError(
                "Job embedding generation returned an empty result."
            )

        if len(embedding) != 384:
            raise ValueError(
                "Expected a 384-dimensional embedding, "
                f"but received {len(embedding)} dimensions."
            )

        # Flush new jobs only after embedding succeeds so the API can persist
        # the job and AI profile together in one transaction.
        if job.id is None:
            db.flush()

        # 5. Find an existing AI profile.
        ai_profile = (
            db.query(JobAIProfile)
            .filter(JobAIProfile.job_id == job.id)
            .first()
        )

        # 6. Create or update the AI profile.
        if ai_profile is None:
            ai_profile = JobAIProfile(
                job_id=job.id,
                required_skills=required_skills,
                preferred_skills=preferred_skills,
                embedding_dimension=len(embedding),
                embedding_reference=None,
                processing_version=PROCESSING_VERSION,
            )

            db.add(ai_profile)

        else:
            ai_profile.required_skills = required_skills
            ai_profile.preferred_skills = preferred_skills
            ai_profile.embedding_dimension = len(embedding)
            ai_profile.processing_version = PROCESSING_VERSION
        ai_profile.processed_at = datetime.utcnow()


        # 7. Persist the profile.
        db.commit()
        db.refresh(ai_profile)

        logger.info(
            "Job processed (job_id=%s, required_skills=%d, "
            "preferred_skills=%d, embedding_dimensions=%d, version=%s)",
            job.id,
            len(required_skills),
            len(preferred_skills),
            len(embedding),
            PROCESSING_VERSION,
        )

        return job

    except Exception:
        db.rollback()
        raise
