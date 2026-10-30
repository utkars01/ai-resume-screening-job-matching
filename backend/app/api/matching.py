from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.database.session import get_db
from backend.app.models.candidate import Candidate
from backend.app.models.job import Job
from backend.app.models.resume import Resume
from backend.app.models.resume_ai_profile import ResumeAIProfile
from backend.app.models.job_ai_profile import JobAIProfile
from backend.app.services.matching_service import matching_service


router = APIRouter(
    prefix="/api/matching",
    tags=["AI Matching"],
)


@router.post("/jobs/{job_id}/rank-candidates")
def rank_candidates(
    job_id: int,
    db: Session = Depends(get_db),
):
    # 1. Validate job.
    job = (
        db.query(Job)
        .filter(Job.id == job_id)
        .first()
    )

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found",
        )

    if job.status != "active":
        raise HTTPException(
            status_code=400,
            detail="Only active jobs can be matched",
        )

    # 2. Retrieve job AI profile.
    job_profile = (
        db.query(JobAIProfile)
        .filter(JobAIProfile.job_id == job_id)
        .first()
    )

    if job_profile is None:
        raise HTTPException(
            status_code=409,
            detail="Job AI profile is missing. Reprocess this job first.",
        )

    candidates = db.query(Candidate).all()

    results = []
    skipped = []

    # 3. Evaluate each candidate.
    for candidate in candidates:
        resumes = (
            db.query(Resume)
            .filter(
                Resume.candidate_id == candidate.id,
                Resume.processing_status == "completed",
            )
            .order_by(
                Resume.created_at.desc(),
                Resume.id.desc(),
            )
            .all()
        )

        if not resumes:
            skipped.append({
                "candidate_id": candidate.id,
                "reason": "No successfully processed resume",
            })
            continue

        # Select the newest resume with an AI profile.
        selected_resume = None
        resume_profile = None

        for resume in resumes:
            profile = (
                db.query(ResumeAIProfile)
                .filter(
                    ResumeAIProfile.resume_id == resume.id
                )
                .first()
            )

            if profile is not None:
                selected_resume = resume
                resume_profile = profile
                break

        if selected_resume is None:
            skipped.append({
                "candidate_id": candidate.id,
                "reason": "No resume AI profile available",
            })
            continue

        try:
            calculated = matching_service.calculate_features(
                candidate=candidate,
                resume=selected_resume,
                resume_profile=resume_profile,
                job=job,
                job_profile=job_profile,
            )

            score = matching_service.predict_score(
                calculated["features"]
            )

            results.append({
                "candidate_id": candidate.id,
                "resume_id": selected_resume.id,
                "candidate_role": candidate.role,

                # Raw model score, not a probability.
                "ranking_score": score,

                "semantic_similarity": (
                    calculated["semantic_similarity"]
                ),
                "skill_coverage": calculated["skill_coverage"],
                "baseline_score": calculated["baseline_score"],

                # Required skills.
                "matched_skills": calculated["matched_skills"],
                "missing_skills": calculated["missing_skills"],

                # Preferred skills.
                "preferred_skill_coverage": (
                    calculated["preferred_skill_coverage"]
                ),
                "matched_preferred_skills": (
                    calculated["matched_preferred_skills"]
                ),
                "missing_preferred_skills": (
                    calculated["missing_preferred_skills"]
                ),

                # Individual compatibility features.
                "role_compatibility": (
                    calculated["features"]["role_compatibility"]
                ),
                "experience_compatibility": (
                    calculated["features"]["experience_compatibility"]
                ),
                "seniority_compatibility": (
                    calculated["features"]["seniority_compatibility"]
                ),
                "education_compatibility": (
                    calculated["features"]["education_compatibility"]
                ),
            })

        except Exception as exc:
            import traceback
            traceback.print_exc()
            # Avoid exposing internal exception details in API responses.
            skipped.append({
                "candidate_id": candidate.id,
                "resume_id": selected_resume.id,
                "reason": (
                    "Candidate matching failed. "
                    "Check application logs for details."
                ),
            })

    # 4. Sort candidates by descending model score.
    results.sort(
        key=lambda item: item["ranking_score"],
        reverse=True,
    )

    for position, result in enumerate(results, start=1):
        result["rank"] = position

    # 5. Return the ranking response.
    return {
        "job_id": job.id,
        "job_title": job.title,
        "ranked_candidate_count": len(results),
        "skipped_candidate_count": len(skipped),
        "ranking_score_type": "raw_xgboost_score_not_probability",
        "results": results,
        "skipped_candidates": skipped,
    }