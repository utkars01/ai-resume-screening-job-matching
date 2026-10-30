from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.app.database.session import get_db
from backend.app.models.candidate import Candidate
from backend.app.models.resume import Resume
from backend.app.models.resume_ai_profile import ResumeAIProfile
from backend.app.schemas.resume_ai_profile import ResumeAIProfileResponse
from backend.app.services.document_parser import extract_text, validate_document_content
from backend.app.services.resume_processor import process_resume


router = APIRouter(
    prefix="/api/resumes",
    tags=["Resumes"],
)


UPLOAD_DIR = Path("data/uploads")

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
}

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB


@router.post("/upload", status_code=201)
async def upload_resume(
    candidate_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    # ---------------------------------------------------------
    # 1. Verify candidate exists
    # ---------------------------------------------------------
    candidate = (
        db.query(Candidate)
        .filter(Candidate.id == candidate_id)
        .first()
    )

    if candidate is None:
        raise HTTPException(
            status_code=404,
            detail="Candidate not found",
        )

    # ---------------------------------------------------------
    # 2. Validate filename
    # ---------------------------------------------------------
    original_filename = file.filename or ""

    if not original_filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    # ---------------------------------------------------------
    # 3. Validate file extension
    # ---------------------------------------------------------
    extension = Path(original_filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Only PDF and DOCX are allowed.",
        )

    # ---------------------------------------------------------
    # 4. Read uploaded file
    # ---------------------------------------------------------
    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    # ---------------------------------------------------------
    # 5. Validate file size
    # ---------------------------------------------------------
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="File size exceeds the 5 MB limit.",
        )

    if not validate_document_content(file_bytes, extension):
        raise HTTPException(
            status_code=400,
            detail="The file content does not match its PDF or DOCX extension.",
        )

    # ---------------------------------------------------------
    # 6. Create upload directory
    # ---------------------------------------------------------
    UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # 7. Generate safe unique filename
    # ---------------------------------------------------------
    stored_filename = f"{uuid4().hex}{extension}"

    file_path = UPLOAD_DIR / stored_filename

    # ---------------------------------------------------------
    # 8. Save file
    # ---------------------------------------------------------
    file_path.write_bytes(file_bytes)

    # ---------------------------------------------------------
    # 9. Extract resume text
    # ---------------------------------------------------------
    try:
        extracted_text = extract_text(
            str(file_path),
            extension,
        )

    except Exception:
        file_path.unlink(
            missing_ok=True
        )

        raise HTTPException(
            status_code=400,
            detail="Unable to extract text from the uploaded document.",
        )

    # ---------------------------------------------------------
    # 10. Validate extracted text
    # ---------------------------------------------------------
    if not extracted_text.strip():
        file_path.unlink(
            missing_ok=True
        )

        raise HTTPException(
            status_code=400,
            detail="No readable text was found in the uploaded document.",
        )

    # ---------------------------------------------------------
    # 11. Create resume database record
    # ---------------------------------------------------------
    resume = Resume(
        candidate_id=candidate_id,
        original_filename=original_filename,
        file_path=str(file_path),
        file_type=extension,
        extracted_text=extracted_text,
        processing_status="pending",
    )

    db.add(resume)
    db.commit()
    db.refresh(resume)

    # ---------------------------------------------------------
    # 12. Process resume through AI pipeline
    # ---------------------------------------------------------
    try:
        resume = process_resume(
            resume=resume,
            db=db,
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Resume uploaded, but AI processing failed.",
        )

    # ---------------------------------------------------------
    # 13. Return response
    # ---------------------------------------------------------
    return {
        "message": "Resume uploaded and processed successfully",
        "resume_id": resume.id,
        "candidate_id": candidate_id,
        "original_filename": original_filename,
        "file_type": extension,
        "processing_status": resume.processing_status,
        "extracted_text_length": len(extracted_text),
    }


@router.get("/")
def get_resumes(
    db: Session = Depends(get_db),
):
    resumes = (
        db.query(Resume)
        .all()
    )

    return [
        {
            "id": resume.id,
            "candidate_id": resume.candidate_id,
            "original_filename": resume.original_filename,
            "file_type": resume.file_type,
            "processing_status": resume.processing_status,
            "extracted_text_length": (
                len(resume.extracted_text)
                if resume.extracted_text
                else 0
            ),
            "created_at": resume.created_at,
        }
        for resume in resumes
    ]


@router.get(
    "/{resume_id}/ai-profile",
    response_model=ResumeAIProfileResponse,
)
def get_resume_ai_profile(
    resume_id: int,
    db: Session = Depends(get_db),
):
    # ---------------------------------------------------------
    # 1. Verify resume exists
    # ---------------------------------------------------------
    resume = (
        db.query(Resume)
        .filter(Resume.id == resume_id)
        .first()
    )

    if resume is None:
        raise HTTPException(
            status_code=404,
            detail="Resume not found",
        )

    # ---------------------------------------------------------
    # 2. Find AI profile
    # ---------------------------------------------------------
    ai_profile = (
        db.query(ResumeAIProfile)
        .filter(
            ResumeAIProfile.resume_id == resume_id
        )
        .first()
    )

    if ai_profile is None:
        raise HTTPException(
            status_code=404,
            detail="AI profile not found for this resume",
        )

    return ai_profile
