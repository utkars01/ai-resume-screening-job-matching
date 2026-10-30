from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.app.schemas.candidate_update import CandidateUpdate
from backend.app.database.session import get_db
from backend.app.models.candidate import Candidate
from backend.app.models.user import User
from backend.app.schemas.candidate import (
    CandidateCreate,
    CandidateResponse,
)


router = APIRouter(
    prefix="/api/candidates",
    tags=["Candidates"],
)


@router.post(
    "/",
    response_model=CandidateResponse,
    status_code=201,
)
def create_candidate(
    candidate_data: CandidateCreate,
    db: Session = Depends(get_db),
):
    user = (
        db.query(User)
        .filter(
            User.id == candidate_data.user_id,
            User.role == "candidate",
        )
        .first()
    )

    if user is None:
        raise HTTPException(
            status_code=400,
            detail="Valid candidate user not found",
        )

    existing_candidate = (
        db.query(Candidate)
        .filter(
            Candidate.user_id == candidate_data.user_id
        )
        .first()
    )

    if existing_candidate:
        raise HTTPException(
            status_code=409,
            detail="Candidate profile already exists for this user",
        )

    candidate = Candidate(
        user_id=candidate_data.user_id,
        phone=candidate_data.phone,
        location=candidate_data.location,
        headline=candidate_data.headline,
        summary=candidate_data.summary,
        role=candidate_data.role,
        seniority=candidate_data.seniority,
        years_experience=candidate_data.years_experience,
        education=candidate_data.education,
    )

    db.add(candidate)
    db.commit()
    db.refresh(candidate)

    return candidate


@router.get(
    "/",
    response_model=list[CandidateResponse],
)
def get_candidates(
    db: Session = Depends(get_db),
):
    return db.query(Candidate).all()


@router.get(
    "/{candidate_id}",
    response_model=CandidateResponse,
)
def get_candidate(
    candidate_id: int,
    db: Session = Depends(get_db),
):
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

    return candidate
@router.patch("/{candidate_id}", response_model=CandidateResponse)
def update_candidate(
    candidate_id: int,
    updates: CandidateUpdate,
    db: Session = Depends(get_db),
):
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

    for field, value in updates.model_dump(exclude_unset=True).items():
        setattr(candidate, field, value)

    db.commit()
    db.refresh(candidate)

    return candidate