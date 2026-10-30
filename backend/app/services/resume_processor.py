from sqlalchemy.orm import Session

from backend.app.models.resume import Resume
from backend.app.models.resume_ai_profile import ResumeAIProfile
from backend.app.services.embedding_service import embedding_service
from backend.app.services.skill_service import skill_service


PROCESSING_VERSION = "v1"


def process_resume(
    resume: Resume,
    db: Session,
) -> Resume:
    """
    Process a stored resume using the AI pipeline.

    Current processing:
    1. Validate extracted resume text
    2. Extract canonical skills
    3. Generate 384-dimensional embedding
    4. Create/update ResumeAIProfile
    5. Mark resume as completed
    """

    if not resume.extracted_text:
        resume.processing_status = "failed"

        db.commit()
        db.refresh(resume)

        return resume

    try:
        # -----------------------------------------------------
        # 1. Mark processing
        # -----------------------------------------------------
        resume.processing_status = "processing"

        db.commit()
        db.refresh(resume)

        # -----------------------------------------------------
        # 2. Extract canonical skills
        # -----------------------------------------------------
        skills = skill_service.extract_skills(
            resume.extracted_text
        )

        # -----------------------------------------------------
        # 3. Generate semantic embedding
        # -----------------------------------------------------
        embedding = embedding_service.generate_embedding(
            resume.extracted_text
        )

        if not embedding:
            raise ValueError(
                "Embedding generation returned an empty result."
            )

        if len(embedding) != 384:
            raise ValueError(
                f"Expected 384-dimensional embedding, "
                f"received {len(embedding)} dimensions."
            )

        # -----------------------------------------------------
        # 4. Find existing AI profile
        # -----------------------------------------------------
        ai_profile = (
            db.query(ResumeAIProfile)
            .filter(
                ResumeAIProfile.resume_id == resume.id
            )
            .first()
        )

        # -----------------------------------------------------
        # 5. Create or update AI profile
        # -----------------------------------------------------
        if ai_profile is None:

            ai_profile = ResumeAIProfile(
                resume_id=resume.id,
                extracted_skills=skills,
                embedding_dimension=len(embedding),
                embedding_reference=None,
                processing_version=PROCESSING_VERSION,
            )

            db.add(ai_profile)

        else:

            ai_profile.extracted_skills = skills
            ai_profile.embedding_dimension = len(embedding)
            ai_profile.processing_version = (
                PROCESSING_VERSION
            )

        # -----------------------------------------------------
        # 6. Mark resume processing as completed
        # -----------------------------------------------------
        resume.processing_status = "completed"

        db.commit()

        db.refresh(resume)
        db.refresh(ai_profile)

        print(
            f"Resume {resume.id} processed successfully."
        )

        print(
            f"Extracted skills: {len(skills)}"
        )

        print(
            f"Embedding dimension: {len(embedding)}"
        )

        return resume

    except Exception:

        resume.processing_status = "failed"

        db.commit()
        db.refresh(resume)

        raise