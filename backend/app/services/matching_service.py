from pathlib import Path

import numpy as np
import pandas as pd
from threading import Lock

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

        self._model = None
        self._model_lock = Lock()

    def _get_model(self):
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    if not MODEL_PATH.exists():
                        raise FileNotFoundError(
                            f"XGBoost model not found: {MODEL_PATH}"
                        )
                    import xgboost as xgb

                    model = xgb.Booster()
                    model.load_model(str(MODEL_PATH))
                    self._model = model
        return self._model

    def calculate_features(
        self,
        candidate: Candidate,
        resume: Resume,
        resume_profile: ResumeAIProfile,
        job: Job,
        job_profile: JobAIProfile,
        job_embedding: list[float] | None = None,
    ) -> dict:
        # Prioritize key qualification signals (role, seniority, experience, top skills)
        # at the start of the semantic text so Sentence Transformers' 256-token limit
        # focuses on relevant qualifications rather than personal headers or boilerplate.
        semantic_resume_parts = []
        if getattr(candidate, "role", None):
            semantic_resume_parts.append(f"Role: {candidate.role}")
        if getattr(candidate, "seniority", None):
            semantic_resume_parts.append(f"Seniority: {candidate.seniority}")
        if getattr(candidate, "years_experience", None) is not None:
            semantic_resume_parts.append(f"Experience: {candidate.years_experience} years")
        extracted_skills = getattr(resume_profile, "extracted_skills", None) or []
        if extracted_skills:
            semantic_resume_parts.append(f"Skills: {', '.join(extracted_skills[:25])}")
        if getattr(candidate, "education", None):
            semantic_resume_parts.append(f"Education: {candidate.education}")

        raw_resume_text = (getattr(resume, "extracted_text", None) or "").strip()
        if raw_resume_text:
            semantic_resume_parts.append(raw_resume_text)

        resume_text = "\n".join(semantic_resume_parts) if semantic_resume_parts else (raw_resume_text or "Candidate")

        if job_embedding is None:
            job_text_parts = [
                f"Title: {job.title}" if getattr(job, "title", None) else "",
                f"Required Skills: {job.required_skills}" if getattr(job, "required_skills", None) else "",
                f"Preferred Skills: {job.preferred_skills}" if getattr(job, "preferred_skills", None) else "",
                f"Experience: {job.experience_required} years" if getattr(job, "experience_required", None) is not None else "",
                getattr(job, "description", None) or "",
            ]
            job_text = "\n".join(
                part.strip() for part in job_text_parts if part and part.strip()
            )
            job_embedding = embedding_service.generate_embedding(job_text)

        resume_embedding = embedding_service.generate_embedding(resume_text)
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

    @staticmethod
    def calculate_calibrated_fit(
        raw_score: float,
        features: dict,
        required_skills: list | None = None,
        matched_required: list | None = None,
        preferred_skills: list | None = None,
        matched_preferred: list | None = None,
        semantic_similarity: float | None = None,
    ) -> dict:
        """
        Calculate a Required-Skill-Dominant candidate evaluation score (0-100%) and tier.

        Required Skills are treated as the primary gatekeeper and dominant ranking factor:
        - Required Skills Match: 55% of the primary score.
        - ML Ranking Model Signal: 15% (sigmoid-scaled XGBoost non-linear interactions).
        - Semantic Content Relevance: 15% (Sentence Transformers cosine similarity).
        - Profile Compatibility: 15% (Role, Experience, Seniority, Education).
        - Preferred Skills Bonus: Up to +5% additive differentiator / tie-breaker.

        Strict Guardrails for Required Skills:
        - 0% Required Skills Match: Hard ceiling cap at max 30.0% fit, preventing unqualified
          candidates from ranking high regardless of semantic buzzwords or model spikes.
        - < 50% Required Skills Match: Strict 25% penalty on composite score.
        - 100% Required Skills Match: 5% reward boost for complete core coverage.
        """
        import math

        req_count = len(required_skills or [])
        matched_count = len(matched_required or [])

        # 1. Primary Dominant Component: Required Skills Match (0.0 to 1.0)
        if req_count > 0:
            required_skill_ratio = matched_count / req_count
        else:
            resume_skills_count = features.get("resume_skill_count", 0)
            required_skill_ratio = min(1.0, resume_skills_count / 5.0)

        # 2. ML Ranker Signal (sigmoid scaled into [0.0, 1.0])
        clamped_score = max(-5.0, min(5.0, float(raw_score)))
        ml_score = 1.0 / (1.0 + math.exp(-2.0 * clamped_score))

        # 3. Semantic Similarity (Sentence Transformers)
        if semantic_similarity is None:
            semantic_similarity = features.get("semantic_similarity", 0.5)
        semantic_sim = max(0.0, min(1.0, float(semantic_similarity)))

        # 4. Profile Compatibility (Role, Experience, Seniority, Education)
        role_comp = float(features.get("role_compatibility", 0.5))
        exp_comp = float(features.get("experience_compatibility", 0.5))
        sen_comp = float(features.get("seniority_compatibility", 0.5))
        edu_comp = float(features.get("education_compatibility", 0.5))
        profile_score = (0.35 * role_comp) + (0.35 * exp_comp) + (0.15 * sen_comp) + (0.15 * edu_comp)

        # 5. Preferred Skills Differentiator (Up to +5% additive bonus)
        pref_count = len(preferred_skills or [])
        matched_pref_count = len(matched_preferred or [])
        pref_bonus = 0.05 * (matched_pref_count / pref_count) if pref_count > 0 else 0.0

        # 6. Dominant Required-Skills Weighting:
        # When required skills are present, they drive 55% of the overall composite score
        if req_count > 0:
            base_composite = (
                (0.55 * required_skill_ratio)
                + (0.15 * ml_score)
                + (0.15 * semantic_sim)
                + (0.15 * profile_score)
                + pref_bonus
            )

            # Strict Required Skill Guardrails:
            if matched_count == 0:
                # Disqualifying ceiling: Cannot score above 30.0% if missing ALL required skills
                base_composite = min(0.30, base_composite * 0.40)
            elif required_skill_ratio < 0.5:
                # Heavy penalty if candidate matches less than half of core required skills
                base_composite *= 0.75
            elif required_skill_ratio == 1.0:
                # Core mastery reward
                base_composite = min(1.0, base_composite * 1.05)
        else:
            # Fallback when job description did not specify explicit required skills
            base_composite = (
                (0.35 * required_skill_ratio)
                + (0.25 * ml_score)
                + (0.20 * semantic_sim)
                + (0.20 * profile_score)
                + pref_bonus
            )

        composite_score = round(max(0.0, min(100.0, base_composite * 100.0)), 1)

        # 7. Categorical match tier
        if composite_score >= 80.0:
            fit_tier = "Strong Match"
        elif composite_score >= 65.0:
            fit_tier = "Good Match"
        elif composite_score >= 50.0:
            fit_tier = "Moderate Match"
        else:
            fit_tier = "Low Match"

        return {
            "composite_score": composite_score,
            "fit_tier": fit_tier,
            "required_skill_ratio": round(required_skill_ratio * 100.0, 1),
            "score_breakdown": {
                "required_skill_pct": round(required_skill_ratio * 100.0, 1),
                "semantic_pct": round(semantic_sim * 100.0, 1),
                "profile_pct": round(profile_score * 100.0, 1),
                "model_signal_pct": round(ml_score * 100.0, 1),
                "preferred_bonus_pct": round(pref_bonus * 100.0, 1),
            },
        }

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

        import xgboost as xgb

        matrix = xgb.DMatrix(feature_frame)

        # This is a raw XGBoost ranking score, NOT a probability.
        return float(self._get_model().predict(matrix)[0])


matching_service = MatchingService()
