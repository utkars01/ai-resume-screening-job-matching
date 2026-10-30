from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ResumeAIProfileResponse(BaseModel):
    id: int
    resume_id: int
    extracted_skills: list[str]
    embedding_dimension: int | None
    embedding_reference: str | None
    processing_version: str
    processed_at: datetime

    model_config = ConfigDict(from_attributes=True)