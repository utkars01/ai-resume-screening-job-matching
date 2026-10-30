from datetime import datetime

from pydantic import BaseModel, ConfigDict


class JobAIProfileResponse(BaseModel):
    id: int
    job_id: int
    required_skills: list[str]
    preferred_skills: list[str] = []
    embedding_dimension: int | None
    embedding_reference: str | None
    processing_version: str
    processed_at: datetime

    model_config = ConfigDict(from_attributes=True)