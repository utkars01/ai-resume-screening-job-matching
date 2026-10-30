from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database.base import Base


class Resume(Base):
    __tablename__ = "resumes"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    candidate_id: Mapped[int] = mapped_column(
        ForeignKey("candidates.id"),
        nullable=False,
        index=True,
    )

    original_filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    file_path: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    file_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    extracted_text: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    processing_status: Mapped[str] = mapped_column(
        String(50),
        default="pending",
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    candidate = relationship("Candidate", backref="resumes")