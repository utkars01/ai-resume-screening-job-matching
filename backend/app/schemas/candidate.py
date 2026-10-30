from pydantic import BaseModel, ConfigDict


class CandidateCreate(BaseModel):
    user_id: int

    phone: str | None = None
    location: str | None = None
    headline: str | None = None
    summary: str | None = None

    role: str | None = None
    seniority: str | None = None
    years_experience: int | None = None
    education: str | None = None


class CandidateResponse(BaseModel):
    id: int
    user_id: int

    phone: str | None
    location: str | None
    headline: str | None
    summary: str | None

    role: str | None
    seniority: str | None
    years_experience: int | None
    education: str | None

    model_config = ConfigDict(from_attributes=True)