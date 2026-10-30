from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database.base import Base


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    recruiter_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )

    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    company_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    location: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    industry: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    required_skills: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    preferred_skills: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    experience_required: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    seniority_required: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    education_required: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        default="active",
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    recruiter = relationship("User", backref="jobs")