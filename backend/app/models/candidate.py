from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database.base import Base


class Candidate(Base):
    __tablename__ = "candidates"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"),
        unique=True,
        nullable=False,
    )

    phone: Mapped[str | None] = mapped_column(
        String(30),
        nullable=True,
    )

    location: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    headline: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    summary: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # Candidate attributes required by the AI ranking model
    role: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    seniority: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    years_experience: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    education: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    user = relationship(
        "User",
        backref="candidate_profile",
    )