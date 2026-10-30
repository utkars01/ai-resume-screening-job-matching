from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database.base import Base


class ApplicantAIProfile(Base):
    __tablename__ = "applicant_ai_profiles"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    applicant_id: Mapped[int] = mapped_column(
        ForeignKey("applicants.id"),
        unique=True,
        nullable=False,
        index=True,
    )

    extracted_skills: Mapped[list] = mapped_column(
        JSON, nullable=False, default=list
    )
    embedding_dimension: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    embedding_reference: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )
    processing_version: Mapped[str] = mapped_column(
        Text, nullable=False, default="v1"
    )
    processed_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    applicant = relationship("Applicant", backref="ai_profile")
