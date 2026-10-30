from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database.base import Base


class Applicant(Base):
    __tablename__ = "applicants"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    # Each screening submission belongs to the job being screened.
    # No User or Candidate account is required.
    job_id: Mapped[int] = mapped_column(
        ForeignKey("jobs.id"),
        nullable=False,
        index=True,
    )

    original_filename: Mapped[str] = mapped_column(
        String(255), nullable=False
    )
    file_path: Mapped[str] = mapped_column(
        String(500), nullable=False
    )
    file_type: Mapped[str] = mapped_column(
        String(20), nullable=False
    )
    extracted_text: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )
    processing_status: Mapped[str] = mapped_column(
        String(50), default="pending", nullable=False
    )

    # Optional profile fields. Unknown information stays NULL.
    role: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )
    seniority: Mapped[str | None] = mapped_column(
        String(100), nullable=True
    )
    years_experience: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    education: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    job = relationship("Job", backref="applicants")
