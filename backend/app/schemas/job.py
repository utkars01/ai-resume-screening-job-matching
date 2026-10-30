from pydantic import BaseModel, ConfigDict, Field


class JobCreate(BaseModel):
    title: str
    company_name: str
    location: str | None = None
    industry: str | None = None
    description: str
    required_skills: str | None = None
    preferred_skills: str | None = None
    experience_required: int | None = Field(default=None, ge=0)
    seniority_required: str | None = None
    education_required: str | None = None


class JobResponse(BaseModel):
    id: int
    recruiter_id: int
    title: str
    company_name: str
    location: str | None
    industry: str | None
    description: str
    required_skills: str | None
    preferred_skills: str | None
    experience_required: int | None
    seniority_required: str | None
    education_required: str | None
    status: str

    model_config = ConfigDict(from_attributes=True)
