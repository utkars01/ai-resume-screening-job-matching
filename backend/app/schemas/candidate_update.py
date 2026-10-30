from pydantic import BaseModel, ConfigDict, Field


class CandidateUpdate(BaseModel):
    phone: str | None = None
    location: str | None = None
    headline: str | None = None
    summary: str | None = None

    role: str | None = None
    seniority: str | None = None
    years_experience: int | None = Field(default=None, ge=0)
    education: str | None = None

    model_config = ConfigDict(extra="forbid")