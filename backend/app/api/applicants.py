
import logging
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.database.session import get_db
from backend.app.auth import get_current_recruiter
from backend.app.models.applicant import Applicant
from backend.app.models.applicant_ai_profile import ApplicantAIProfile
from backend.app.models.job import Job
from backend.app.models.job_ai_profile import JobAIProfile
from backend.app.models.user import User
from backend.app.services.document_parser import extract_text, validate_document_content
from backend.app.services.embedding_service import embedding_service
from backend.app.services.matching_service import matching_service
from backend.app.services.applicant_metadata import extract_applicant_metadata
from backend.app.services.skill_service import skill_service
from backend.app.config import settings


logger = logging.getLogger(__name__)


def _build_ranking_reason(result: dict, rank: int, all_results: list) -> str:
    """Generate a plain-English explanation of why this candidate is ranked here."""
    reasons: list[str] = []

    req_pct = round(result.get("skill_coverage", 0) * 100)
    matched = result.get("matched_skills") or []
    missing = result.get("missing_skills") or []
    total_req = len(matched) + len(missing)
    role_comp = result.get("role_compatibility") or 0
    exp_comp = result.get("experience_compatibility") or 0
    composite = result.get("composite_score") or 0
    match_tier = result.get("match_tier") or ""

    # --- Primary driver: required skill coverage ---
    if total_req == 0:
        reasons.append("No required skills listed for comparison")
    elif req_pct == 100:
        reasons.append(f"Matches all {len(matched)} required skill{'s' if len(matched) != 1 else ''}")
    elif req_pct >= 75:
        reasons.append(
            f"Strong required skill coverage — {req_pct}% ({len(matched)} of {total_req} skills matched)"
        )
    elif req_pct >= 50:
        reasons.append(
            f"Adequate required skill coverage — {req_pct}% ({len(matched)} of {total_req} skills matched)"
        )
    else:
        reasons.append(
            f"Low required skill coverage — only {req_pct}% ({len(missing)} of {total_req} core skills missing)"
        )

    # --- Secondary signals ---
    if role_comp >= 0.80:
        reasons.append("strong role alignment")
    elif role_comp >= 0.60:
        reasons.append("moderate role alignment")

    if exp_comp >= 0.80:
        reasons.append("meets experience requirements")
    elif exp_comp < 0.40:
        reasons.append("below experience requirement")

    if match_tier in ("Strong Match", "Good Match"):
        reasons.append(f"overall {match_tier.lower()}")

    # --- Peer comparison (ranks 2+) ---
    if rank > 1 and len(all_results) > 1:
        above = all_results[rank - 2]  # zero-indexed entry ranked just above
        above_req_pct = round(above.get("skill_coverage", 0) * 100)
        above_composite = above.get("composite_score") or 0

        if req_pct > above_req_pct:
            reasons.append(
                f"ranked above #{rank - 1} due to higher required skill coverage "
                f"({req_pct}% vs {above_req_pct}%)"
            )
        elif req_pct < above_req_pct:
            reasons.append(
                f"ranked below #{rank - 1} who covers {above_req_pct}% of required skills "
                f"vs this candidate's {req_pct}%"
            )
        elif composite >= above_composite - 0.5:
            reasons.append(
                f"comparable overall fit to #{rank - 1} — role and experience signals "
                "differentiate the final order"
            )

    return (reasons[0].capitalize() + ("; " + "; ".join(reasons[1:]) if len(reasons) > 1 else "") + ".")

router = APIRouter(
    prefix="/api/applicants",
    tags=["Applicants"],
)

UPLOAD_DIR = settings.upload_storage_dir
ALLOWED_EXTENSIONS = {".pdf", ".docx"}
MAX_FILE_SIZE = 5 * 1024 * 1024
MAX_FILES_PER_BATCH = 50
PROCESSING_VERSION = "v1"


def _applicant_summary(applicant: Applicant) -> dict:
    return {
        "id": applicant.id,
        "job_id": applicant.job_id,
        "filename": applicant.original_filename,
        "file_type": applicant.file_type,
        "processing_status": applicant.processing_status,
        "role": applicant.role,
        "years_experience": applicant.years_experience,
        "seniority": applicant.seniority,
        "education": applicant.education,
        "created_at": applicant.created_at,
    }


def _require_schema_tables(db: Session, required: set[str]) -> None:
    rows = db.execute(
        text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = current_schema() AND table_name IN "
            "('screening_runs', 'screening_results', 'applicant_shortlists')"
        )
    ).scalars().all()
    missing = required - set(rows)
    if missing:
        raise HTTPException(
            status_code=503,
            detail="Screening history and shortlists require the reviewed database schema update.",
        )


class ShortlistRequest(BaseModel):
    note: str | None = Field(default=None, max_length=2000)


@router.get("/")
def list_recruiter_applicants(
    db: Session = Depends(get_db),
    recruiter: User = Depends(get_current_recruiter),
):
    applicants = (
        db.query(Applicant)
        .join(Job, Applicant.job_id == Job.id)
        .filter(Job.recruiter_id == recruiter.id)
        .order_by(Applicant.created_at.desc(), Applicant.id.desc())
        .all()
    )
    return [_applicant_summary(applicant) for applicant in applicants]


@router.get("/jobs/{job_id}")
def list_job_applicants(
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
        raise HTTPException(status_code=404, detail="Job not found.")

    applicants = (
        db.query(Applicant)
        .filter(Applicant.job_id == job_id)
        .order_by(Applicant.created_at.desc(), Applicant.id.desc())
        .all()
    )
    return [_applicant_summary(applicant) for applicant in applicants]


@router.post("/bulk-upload", status_code=201)
async def bulk_upload_applicants(
    files: list[UploadFile] = File(...),
    job_id: int = Query(..., gt=0),
    db: Session = Depends(get_db),
    recruiter: User = Depends(get_current_recruiter),
):
    """Upload and process multiple applicant resumes for one active job."""

    # 1. Validate the selected job.
    job = (
        db.query(Job)
        .filter(Job.id == job_id, Job.recruiter_id == recruiter.id)
        .first()
    )

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found.",
        )

    if job.status != "active":
        raise HTTPException(
            status_code=400,
            detail="Only active jobs can receive screening submissions.",
        )

    # 2. Ensure the job has already been processed for AI matching.
    job_profile = (
        db.query(JobAIProfile)
        .filter(JobAIProfile.job_id == job_id)
        .first()
    )

    if job_profile is None:
        raise HTTPException(
            status_code=409,
            detail="This job has no AI profile. Process or reprocess the job first.",
        )

    if not files:
        raise HTTPException(
            status_code=400,
            detail="Select at least one resume.",
        )

    if len(files) > MAX_FILES_PER_BATCH:
        raise HTTPException(
            status_code=400,
            detail=f"A maximum of {MAX_FILES_PER_BATCH} files is allowed per upload.",
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    seen_upload_hashes = set()

    # 3. Process each resume independently.
    for upload in files:
        original_filename = Path(upload.filename or "").name
        extension = Path(original_filename).suffix.lower()

        file_path = None
        applicant = None

        try:
            if not original_filename:
                raise ValueError("Filename is required.")

            if extension not in ALLOWED_EXTENSIONS:
                raise ValueError("Only PDF and DOCX files are allowed.")

            # Read only one byte beyond the configured limit so a large upload
            # cannot be fully buffered in application memory before rejection.
            file_bytes = await upload.read(MAX_FILE_SIZE + 1)

            if not file_bytes:
                raise ValueError("The uploaded file is empty.")

            if len(file_bytes) > MAX_FILE_SIZE:
                raise ValueError("File exceeds the 5 MB limit.")

            if not validate_document_content(file_bytes, extension):
                raise ValueError("The file content does not match its PDF or DOCX extension.")

            content_hash = hashlib.sha256(file_bytes).hexdigest()
            if content_hash in seen_upload_hashes:
                raise ValueError("This file is duplicated within the selected batch.")

            # Use a unique stored filename to prevent collisions.
            stored_filename = f"{uuid4().hex}{extension}"
            file_path = UPLOAD_DIR / stored_filename
            file_path.write_bytes(file_bytes)

            # 4. Extract document text.
            extracted_text = extract_text(str(file_path), extension)

            if not extracted_text or not extracted_text.strip():
                if extension == ".pdf":
                    raise ValueError(
                        "No selectable text was found in this PDF. It may be "
                        "image-only; OCR is not enabled. Upload a text-based "
                        "PDF or DOCX."
                    )
                raise ValueError("No readable text was found in the document.")

            # 5. Extract metadata BEFORE creating the applicant record.
            metadata = extract_applicant_metadata(extracted_text)

            # 6. Create an applicant without requiring a user account.
            applicant = Applicant(
                job_id=job_id,
                original_filename=original_filename[:255],
                file_path=str(file_path),
                file_type=extension,
                extracted_text=extracted_text,
                processing_status="processing",
                role=metadata["role"],
                years_experience=metadata["years_experience"],
                seniority=metadata["seniority"],
                education=metadata["education"],
            )

            db.add(applicant)
            db.commit()
            db.refresh(applicant)

            seen_upload_hashes.add(content_hash)

            # 7. Reuse the existing AI skill and embedding services.
            skills = skill_service.extract_skills(extracted_text)
            embedding = embedding_service.generate_embedding(extracted_text)

            if embedding is None or len(embedding) != 384:
                raise ValueError(
                    "Embedding generation failed or returned an unexpected dimension."
                )

            # 8. Save the AI profile.
            profile = ApplicantAIProfile(
                applicant_id=applicant.id,
                extracted_skills=skills,
                embedding_dimension=len(embedding),
                embedding_reference=None,
                processing_version=PROCESSING_VERSION,
            )

            db.add(profile)

            applicant.processing_status = "completed"

            db.commit()
            db.refresh(applicant)

            results.append(
                {
                    "filename": original_filename,
                    "applicant_id": applicant.id,
                    "status": "completed",
                    "role": applicant.role,
                    "years_experience": applicant.years_experience,
                    "seniority": applicant.seniority,
                    "education": applicant.education,
                    "extracted_skill_count": len(skills),
                }
            )

        except ValueError as exc:
            db.rollback()

            if applicant is not None and applicant.id is not None:
                saved_applicant = (
                    db.query(Applicant)
                    .filter(Applicant.id == applicant.id)
                    .first()
                )

                if saved_applicant is not None:
                    saved_applicant.processing_status = "failed"
                    db.commit()

                results.append(
                    {
                        "filename": original_filename,
                        "applicant_id": applicant.id,
                        "status": "failed",
                        "message": str(exc),
                    }
                )
            else:
                if file_path is not None:
                    file_path.unlink(missing_ok=True)

                results.append(
                    {
                        "filename": original_filename or "(unnamed file)",
                        "status": "failed",
                        "message": str(exc),
                    }
                )

        except Exception:
            db.rollback()

            logger.exception(
                "Failed to process applicant resume (job_id=%s, applicant_id=%s)",
                job_id,
                getattr(applicant, "id", None),
            )

            if applicant is not None and applicant.id is not None:
                saved_applicant = (
                    db.query(Applicant)
                    .filter(Applicant.id == applicant.id)
                    .first()
                )

                if saved_applicant is not None:
                    saved_applicant.processing_status = "failed"
                    db.commit()

                results.append(
                    {
                        "filename": original_filename,
                        "applicant_id": applicant.id,
                        "status": "failed",
                        "message": "Unexpected processing error. Check backend logs.",
                    }
                )
            else:
                if file_path is not None:
                    file_path.unlink(missing_ok=True)

                results.append(
                    {
                        "filename": original_filename or "(unnamed file)",
                        "status": "failed",
                        "message": "Could not save or process this document.",
                    }
                )

        finally:
            await upload.close()

    completed = sum(
        item["status"] == "completed"
        for item in results
    )

    return {
        "job_id": job_id,
        "total_files": len(results),
        "completed": completed,
        "failed": len(results) - completed,
        "results": results,
    }


@router.post("/jobs/{job_id}/rank")
def rank_applicants(
    job_id: int,
    db: Session = Depends(get_db),
    recruiter: User = Depends(get_current_recruiter),
):
    """Rank processed applicants for a job using the existing XGBoost model."""

    # 1. Validate the job.
    job = (
        db.query(Job)
        .filter(Job.id == job_id, Job.recruiter_id == recruiter.id)
        .first()
    )

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found.",
        )

    if job.status != "active":
        raise HTTPException(
            status_code=400,
            detail="Only active jobs can be matched.",
        )

    # 2. Retrieve the job's AI profile.
    job_profile = (
        db.query(JobAIProfile)
        .filter(JobAIProfile.job_id == job_id)
        .first()
    )

    if job_profile is None:
        raise HTTPException(
            status_code=409,
            detail="Job AI profile is missing.",
        )

    _require_schema_tables(db, {"screening_runs", "screening_results"})

    # 3. Retrieve successfully processed applicants for this job.
    applicants = (
        db.query(Applicant)
        .filter(
            Applicant.job_id == job_id,
            Applicant.processing_status == "completed",
        )
        .order_by(Applicant.id.asc())
        .all()
    )

    results = []
    skipped = []

    # Pre-generate job embedding once for the batch to eliminate redundant inference
    job_embedding = None
    try:
        job_text_parts = [
            f"Title: {job.title}" if getattr(job, "title", None) else "",
            f"Required Skills: {job.required_skills}" if getattr(job, "required_skills", None) else "",
            f"Preferred Skills: {job.preferred_skills}" if getattr(job, "preferred_skills", None) else "",
            f"Experience: {job.experience_required} years" if getattr(job, "experience_required", None) is not None else "",
            getattr(job, "description", None) or "",
        ]
        job_text = "\n".join(part.strip() for part in job_text_parts if part and part.strip())
        if job_text.strip():
            job_embedding = embedding_service.generate_embedding(job_text)
    except Exception:
        logger.warning("Could not pre-generate batch job embedding; falling back to per-applicant embedding.")

    # 4. Calculate features and predict ranking scores.
    for applicant in applicants:
        profile = (
            db.query(ApplicantAIProfile)
            .filter(
                ApplicantAIProfile.applicant_id == applicant.id
            )
            .first()
        )

        if profile is None:
            skipped.append(
                {
                    "applicant_id": applicant.id,
                    "filename": applicant.original_filename,
                    "reason": "Applicant AI profile is missing.",
                }
            )
            continue

        try:
            calculated = matching_service.calculate_features(
                candidate=applicant,
                resume=applicant,
                resume_profile=profile,
                job=job,
                job_profile=job_profile,
                job_embedding=job_embedding,
            )

            score = matching_service.predict_score(
                calculated["features"]
            )

            features = calculated["features"]

            # Compute balanced calibrated fit to smooth out raw XGBoost margin spikes
            calibrated = matching_service.calculate_calibrated_fit(
                raw_score=score,
                features=features,
                required_skills=getattr(job_profile, "required_skills", None),
                matched_required=calculated.get("matched_skills"),
                preferred_skills=getattr(job_profile, "preferred_skills", None),
                matched_preferred=calculated.get("matched_preferred_skills"),
                semantic_similarity=calculated.get("semantic_similarity"),
            )

            results.append(
                {
                    "applicant_id": applicant.id,
                    "filename": applicant.original_filename,
                    "role": applicant.role,
                    "years_experience": applicant.years_experience,
                    "seniority": applicant.seniority,
                    "education": applicant.education,
                    "ranking_score": score,
                    "composite_score": calibrated["composite_score"],
                    "match_tier": calibrated["fit_tier"],
                    "score_breakdown": calibrated["score_breakdown"],
                    "semantic_similarity": calculated[
                        "semantic_similarity"
                    ],
                    "skill_coverage": calculated["skill_coverage"],
                    "baseline_score": calculated["baseline_score"],
                    "matched_skills": calculated["matched_skills"],
                    "missing_skills": calculated["missing_skills"],
                    "preferred_skill_coverage": calculated[
                        "preferred_skill_coverage"
                    ],
                    "matched_preferred_skills": calculated[
                        "matched_preferred_skills"
                    ],
                    "missing_preferred_skills": calculated[
                        "missing_preferred_skills"
                    ],
                    "role_compatibility": features[
                        "role_compatibility"
                    ],
                    "experience_compatibility": features[
                        "experience_compatibility"
                    ],
                    "seniority_compatibility": features[
                        "seniority_compatibility"
                    ],
                    "education_compatibility": features[
                        "education_compatibility"
                    ],
                }
            )

        except Exception:
            logger.exception(
                "Ranking failed for applicant %s",
                applicant.id,
            )

            skipped.append(
                {
                    "applicant_id": applicant.id,
                    "filename": applicant.original_filename,
                    "reason": "Ranking failed; check backend logs.",
                }
            )

    # 5. Rank candidates dominantly based on Required Skill fulfillment and composite fit.
    # Primary sort: composite_score (55%+ driven by required skills with strict guardrails)
    # Secondary sort: direct required skill coverage percentage (skill_coverage)
    # Tertiary sort: raw XGBoost ranking_score
    results.sort(
        key=lambda item: (
            item.get("composite_score", 0.0),
            item.get("skill_coverage", 0.0),
            item["ranking_score"],
        ),
        reverse=True,
    )

    for rank, result in enumerate(results, start=1):
        result["rank"] = rank
        result["ranking_reason"] = _build_ranking_reason(result, rank, results)

    try:
        run_id = db.execute(
            text(
                "INSERT INTO screening_runs (job_id, recruiter_id) "
                "VALUES (:job_id, :recruiter_id) RETURNING id"
            ),
            {"job_id": job.id, "recruiter_id": recruiter.id},
        ).scalar_one()

        evidence_fields = (
            "filename",
            "role",
            "years_experience",
            "seniority",
            "education",
            "composite_score",
            "match_tier",
            "score_breakdown",
            "semantic_similarity",
            "skill_coverage",
            "baseline_score",
            "matched_skills",
            "missing_skills",
            "preferred_skill_coverage",
            "matched_preferred_skills",
            "missing_preferred_skills",
            "role_compatibility",
            "experience_compatibility",
            "seniority_compatibility",
            "education_compatibility",
            "ranking_reason",
        )
        insert_result = text(
            "INSERT INTO screening_results "
            "(screening_run_id, job_id, applicant_id, rank, ranking_score, evidence) "
            "VALUES (:run_id, :job_id, :applicant_id, :rank, :score, CAST(:evidence AS jsonb))"
        )
        for item in results:
            evidence = {
                key: item.get(key)
                for key in evidence_fields
            }
            db.execute(
                insert_result,
                {
                    "run_id": run_id,
                    "job_id": job.id,
                    "applicant_id": item["applicant_id"],
                    "rank": item["rank"],
                    "score": item["ranking_score"],
                    "evidence": json.dumps(evidence),
                },
            )
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Unable to persist screening results (job_id=%s)", job_id)
        raise HTTPException(
            status_code=500,
            detail="Ranking completed, but the screening result could not be saved.",
        )

    return {
        "job_id": job.id,
        "job_title": job.title,
        "screening_run_id": run_id,
        "ranked_applicant_count": len(results),
        "skipped_applicant_count": len(skipped),
        "ranking_score_type": "raw_xgboost_score_not_probability",
        "results": results,
        "skipped_applicants": skipped,
    }


@router.get("/jobs/{job_id}/screening-runs")
def list_screening_runs(
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
        raise HTTPException(status_code=404, detail="Job not found.")
    _require_schema_tables(db, {"screening_runs", "screening_results"})

    rows = db.execute(
        text(
            "SELECT r.id, r.job_id, r.created_at, count(s.id) AS result_count "
            "FROM screening_runs r LEFT JOIN screening_results s "
            "ON s.screening_run_id = r.id AND s.job_id = r.job_id "
            "WHERE r.job_id = :job_id AND r.recruiter_id = :recruiter_id "
            "GROUP BY r.id, r.job_id, r.created_at "
            "ORDER BY r.created_at DESC, r.id DESC"
        ),
        {"job_id": job_id, "recruiter_id": recruiter.id},
    ).mappings().all()
    return [dict(row) for row in rows]


@router.get("/screening-runs/{run_id}")
def get_screening_run(
    run_id: int,
    db: Session = Depends(get_db),
    recruiter: User = Depends(get_current_recruiter),
):
    _require_schema_tables(db, {"screening_runs", "screening_results"})
    run = db.execute(
        text(
            "SELECT id, job_id, created_at FROM screening_runs "
            "WHERE id = :run_id AND recruiter_id = :recruiter_id"
        ),
        {"run_id": run_id, "recruiter_id": recruiter.id},
    ).mappings().first()
    if run is None:
        raise HTTPException(status_code=404, detail="Screening run not found.")

    results = db.execute(
        text(
            "SELECT applicant_id, rank, ranking_score, evidence "
            "FROM screening_results WHERE screening_run_id = :run_id "
            "AND job_id = :job_id ORDER BY rank"
        ),
        {"run_id": run_id, "job_id": run["job_id"]},
    ).mappings().all()
    return {**dict(run), "results": [dict(row) for row in results]}


@router.get("/jobs/{job_id}/shortlist")
def list_shortlisted_applicants(
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
        raise HTTPException(status_code=404, detail="Job not found.")
    _require_schema_tables(db, {"applicant_shortlists"})

    rows = db.execute(
        text(
            "SELECT s.id, s.job_id, s.applicant_id, s.note, s.created_at, "
            "a.original_filename AS filename, a.role, a.years_experience, "
            "a.seniority, a.education, a.processing_status "
            "FROM applicant_shortlists s JOIN applicants a "
            "ON a.job_id = s.job_id AND a.id = s.applicant_id "
            "WHERE s.job_id = :job_id AND s.recruiter_id = :recruiter_id "
            "ORDER BY s.created_at DESC, s.id DESC"
        ),
        {"job_id": job_id, "recruiter_id": recruiter.id},
    ).mappings().all()
    return [dict(row) for row in rows]


@router.post("/jobs/{job_id}/applicants/{applicant_id}/shortlist", status_code=201)
def shortlist_applicant(
    job_id: int,
    applicant_id: int,
    data: ShortlistRequest,
    db: Session = Depends(get_db),
    recruiter: User = Depends(get_current_recruiter),
):
    job = (
        db.query(Job)
        .filter(Job.id == job_id, Job.recruiter_id == recruiter.id)
        .first()
    )
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    applicant = (
        db.query(Applicant)
        .filter(Applicant.id == applicant_id, Applicant.job_id == job_id)
        .first()
    )
    if applicant is None:
        raise HTTPException(status_code=404, detail="Applicant not found for this job.")
    _require_schema_tables(db, {"applicant_shortlists"})

    try:
        row = db.execute(
            text(
                "INSERT INTO applicant_shortlists "
                "(recruiter_id, job_id, applicant_id, note) "
                "VALUES (:recruiter_id, :job_id, :applicant_id, :note) "
                "RETURNING id, recruiter_id, job_id, applicant_id, note, created_at"
            ),
            {
                "recruiter_id": recruiter.id,
                "job_id": job_id,
                "applicant_id": applicant_id,
                "note": data.note,
            },
        ).mappings().one()
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="Applicant is already shortlisted.")
    return dict(row)


@router.delete("/jobs/{job_id}/applicants/{applicant_id}/shortlist", status_code=204)
def remove_shortlisted_applicant(
    job_id: int,
    applicant_id: int,
    db: Session = Depends(get_db),
    recruiter: User = Depends(get_current_recruiter),
):
    job = (
        db.query(Job)
        .filter(Job.id == job_id, Job.recruiter_id == recruiter.id)
        .first()
    )
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    _require_schema_tables(db, {"applicant_shortlists"})

    result = db.execute(
        text(
            "DELETE FROM applicant_shortlists "
            "WHERE recruiter_id = :recruiter_id AND job_id = :job_id "
            "AND applicant_id = :applicant_id"
        ),
        {
            "recruiter_id": recruiter.id,
            "job_id": job_id,
            "applicant_id": applicant_id,
        },
    )
    if result.rowcount != 1:
        db.rollback()
        raise HTTPException(status_code=404, detail="Shortlist entry not found.")
    db.commit()
