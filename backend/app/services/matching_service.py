from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from backend.app.models.candidate import Candidate
from backend.app.models.job import Job
from backend.app.models.resume import Resume
from backend.app.models.resume_ai_profile import ResumeAIProfile
from backend.app.models.job_ai_profile import JobAIProfile

from backend.app.services.embedding_service import embedding_service
from backend.app.services.ranking_feature_contract import (
    MODEL_FEATURE_NAMES,
    build_ranking_feature_components,
    education_compatibility,
    experience_compatibility,
    normalize_text,
    role_compatibility,
    seniority_compatibility,
)


ROOT_DIR = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT_DIR / "models" / "xgboost_ranker.json"
FEATURE_PATH = ROOT_DIR / "models" / "ranking_features.txt"


def cosine_similarity(embedding_a, embedding_b) -> float:
    """Calculate cosine similarity safely for two embeddings."""

    vector_a = np.asarray(embedding_a, dtype=np.float32).reshape(-1)
    vector_b = np.asarray(embedding_b, dtype=np.float32).reshape(-1)

    if vector_a.size == 0 or vector_b.size == 0:
        raise ValueError("Cannot compare empty embeddings.")

    if vector_a.shape != vector_b.shape:
        raise ValueError(
            "Resume and job embeddings have different dimensions."
        )

    if not np.isfinite(vector_a).all() or not np.isfinite(vector_b).all():
        raise ValueError("Embeddings contain non-finite values.")

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    similarity = float(
        np.dot(vector_a, vector_b) / (norm_a * norm_b)
    )

    return float(np.clip(similarity, -1.0, 1.0))


class MatchingService:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"XGBoost model not found: {MODEL_PATH}"
            )

        if not FEATURE_PATH.exists():
            raise FileNotFoundError(
                f"Ranking feature file not found: {FEATURE_PATH}"
            )

        self.model = xgb.Booster()
        self.model.load_model(str(MODEL_PATH))

        self.feature_columns = [
            line.strip()
            for line in FEATURE_PATH.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        if tuple(self.feature_columns) != MODEL_FEATURE_NAMES:
            raise ValueError(
                "The ordered ranking features do not match the "
                "production training contract in ranking_feature_contract.py."
            )

    def calculate_features(
        self,
        candidate: Candidate,
        resume: Resume,
        resume_profile: ResumeAIProfile,
        job: Job,
        job_profile: JobAIProfile,
    ) -> dict:
        resume_text = resume.extracted_text or ""
        job_text_parts = [
            job.title or "",
            job.description or "",
            job.required_skills or "",
            job.preferred_skills or "",
        ]
        job_text = "\n".join(
            part.strip() for part in job_text_parts if part and part.strip()
        )

        resume_embedding = embedding_service.generate_embedding(resume_text)
        job_embedding = embedding_service.generate_embedding(job_text)
        semantic_similarity = cosine_similarity(resume_embedding, job_embedding)

        return build_ranking_feature_components(
            semantic_similarity=semantic_similarity,
            resume_skills=resume_profile.extracted_skills,
            required_skills=job_profile.required_skills,
            preferred_skills=job_profile.preferred_skills,
            resume_role=candidate.role,
            job_title=job.title,
            resume_years=candidate.years_experience,
            required_years=job.experience_required,
            resume_seniority=candidate.seniority,
            job_seniority=job.seniority_required,
            resume_education=candidate.education,
            job_education=job.education_required,
        )

    def predict_score(self, features: dict) -> float:
        missing_features = [
            name
            for name in self.feature_columns
            if name not in features
        ]

        if missing_features:
            raise ValueError(
                f"Missing ranking features: {missing_features}"
            )

        feature_frame = pd.DataFrame(
            [[features[name] for name in self.feature_columns]],
            columns=self.feature_columns,
        )

        feature_frame = feature_frame.apply(
            pd.to_numeric,
            errors="raise",
        )

        if not np.isfinite(feature_frame.to_numpy()).all():
            raise ValueError(
                "Ranking features contain non-finite values."
            )

        matrix = xgb.DMatrix(feature_frame)

        # This is a raw XGBoost ranking score, NOT a probability.
        return float(self.model.predict(matrix)[0])


matching_service = MatchingService()
