
import argparse
import ast
import logging
from pathlib import Path
import time

import numpy as np
import pandas as pd

from backend.app.services.ranking_feature_contract import (
    MODEL_FEATURE_NAMES,
    build_ranking_feature_components,
    normalize_text,
)
from backend.app.services.skill_service import skill_service


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path("data/processed")

BASELINE_FILE = BASE_DIR / "baseline_matching_results.csv"
RESUMES_FILE = BASE_DIR / "resumes_processed.csv"
JOBS_FILE = BASE_DIR / "job_postings_processed.csv"

RESUME_PROFILES_FILE = BASE_DIR / "resume_skill_profiles.csv"
JOB_PROFILES_FILE = BASE_DIR / "job_skill_profiles.csv"

OUTPUT_FILE = BASE_DIR / "ranking_features.csv"
SERIES_PROGRESS_INTERVAL_SECONDS = 60.0
FEATURE_CHUNK_SIZE = 5_000


def normalize_id(series: pd.Series) -> pd.Series:
    """Normalize IDs for case-insensitive joins."""
    return series.astype("string").str.strip().str.upper()


def normalize_skill(value) -> str:
    """Normalize a skill label without changing its meaning."""
    return normalize_text(value)


def parse_list(value):
    """
    Parse Python-list strings, pipe-separated skill profiles,
    or comma-separated values into normalized lists.
    """
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        return sorted(
            {
                normalize_skill(item)
                for item in value
                if normalize_skill(item)
            }
        )

    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    value = str(value).strip()

    if not value:
        return []

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, (list, tuple, set)):
            return sorted(
                {
                    normalize_skill(item)
                    for item in parsed
                    if normalize_skill(item)
                }
            )
    except (ValueError, SyntaxError):
        pass

    # Skill profile files use pipes; processed datasets may use commas.
    separator = "|" if "|" in value else ","

    return sorted(
        {
            normalize_skill(item)
            for item in value.split(separator)
            if normalize_skill(item)
        }
    )


def apply_series_with_progress(series, function, stage):
    """Apply a scalar function with elapsed-time and item progress logs."""
    started = time.perf_counter()
    last_report = started
    total = len(series)
    results = []

    for completed, value in enumerate(series.array, start=1):
        results.append(function(value))
        now = time.perf_counter()
        if (
            completed == 1
            or completed == total
            or now - last_report >= SERIES_PROGRESS_INTERVAL_SECONDS
        ):
            elapsed = now - started
            rate = completed / elapsed if elapsed else 0.0
            logger.info(
                "%s: %s/%s (%.1f%%), %.1fs elapsed, %.2f items/s",
                stage,
                completed,
                total,
                100 * completed / total if total else 100.0,
                elapsed,
                rate,
            )
            last_report = now

    return pd.Series(results, index=series.index, dtype=object)


def parse_skill_fields(features):
    """Parse each resume/job profile once and reuse it for all matching pairs."""
    pair_count = len(features)

    unique_resumes = features[
        ["resume_id", "candidate_skills"]
    ].drop_duplicates("resume_id")
    logger.info(
        "Parsing resume skills: %s unique resumes across %s pairs",
        len(unique_resumes),
        pair_count,
    )
    resume_values = apply_series_with_progress(
        unique_resumes["candidate_skills"],
        parse_list,
        "Resume skill parsing",
    )
    resume_map = pd.Series(
        resume_values.to_list(),
        index=unique_resumes["resume_id"].to_list(),
        dtype=object,
    )
    features["resume_skills_list"] = features["resume_id"].map(resume_map)

    unique_jobs = features[
        ["job_id", "required_skills"]
    ].drop_duplicates("job_id")
    logger.info(
        "Parsing required skills: %s unique jobs across %s pairs",
        len(unique_jobs),
        pair_count,
    )
    required_values = apply_series_with_progress(
        unique_jobs["required_skills"],
        parse_list,
        "Required skill parsing",
    )
    required_map = pd.Series(
        required_values.to_list(),
        index=unique_jobs["job_id"].to_list(),
        dtype=object,
    )
    features["required_skills_list"] = features["job_id"].map(required_map)

    preferred_text = features["preferred_qualifications"].fillna("").astype(str)
    unique_qualifications = preferred_text.drop_duplicates().reset_index(drop=True)
    logger.info(
        "Extracting preferred skills: %s unique qualification texts across %s pairs",
        len(unique_qualifications),
        pair_count,
    )
    preferred_values = apply_series_with_progress(
        unique_qualifications,
        skill_service.extract_skills,
        "Preferred skill extraction",
    )
    preferred_map = pd.Series(
        preferred_values.to_list(),
        index=unique_qualifications.to_list(),
        dtype=object,
    )
    features["preferred_skills_list"] = preferred_text.map(preferred_map)

    for column in (
        "resume_skills_list",
        "required_skills_list",
        "preferred_skills_list",
    ):
        if features[column].isna().any():
            raise ValueError(f"Skill parsing left missing values in {column}")

    return features


def build_model_features_with_progress(features):
    """Build the row-wise feature contract in chunks with visible progress."""
    started = time.perf_counter()
    pieces = []
    total = len(features)

    for start in range(0, total, FEATURE_CHUNK_SIZE):
        stop = min(start + FEATURE_CHUNK_SIZE, total)
        piece = features.iloc[start:stop].apply(
            calculate_skill_features,
            axis=1,
        )
        pieces.append(piece)
        elapsed = time.perf_counter() - started
        completed = stop
        logger.info(
            "Feature vector construction: %s/%s (%.1f%%), %.1fs elapsed",
            completed,
            total,
            100 * completed / total if total else 100.0,
            elapsed,
        )

    if not pieces:
        return pd.DataFrame(index=features.index)

    return pd.concat(pieces)


def calculate_skill_features(row):
    """Use the same ordered 21-feature contract as production inference."""
    components = build_ranking_feature_components(
        semantic_similarity=row["semantic_similarity"],
        resume_skills=row["resume_skills_list"],
        required_skills=row["required_skills_list"],
        preferred_skills=row["preferred_skills_list"],
        resume_role=row["role"],
        job_title=row["normalized_title"],
        resume_years=row["years_experience"],
        required_years=row["years_experience_numeric"],
        resume_seniority=row["seniority"],
        job_seniority=row["experience_level"],
        resume_education=row["education"],
        job_education=row["education_level"],
    )
    return components["features"]


def validate_columns(dataframe, required_columns, dataset_name):
    """Raise a clear error if required columns are missing."""
    missing = set(required_columns) - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"{dataset_name} is missing columns: {sorted(missing)}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Build validated job-resume ranking features."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help=f"CSV destination (default: {OUTPUT_FILE})",
    )
    output_file = parser.parse_args().output

    required_files = [
        BASELINE_FILE,
        RESUMES_FILE,
        JOBS_FILE,
        RESUME_PROFILES_FILE,
        JOB_PROFILES_FILE,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    logger.info("Loading datasets...")

    matching = pd.read_csv(BASELINE_FILE, low_memory=False)
    resumes = pd.read_csv(RESUMES_FILE, low_memory=False)
    jobs = pd.read_csv(JOBS_FILE, low_memory=False)

    resume_profiles = pd.read_csv(
        RESUME_PROFILES_FILE,
        low_memory=False,
    )
    job_profiles = pd.read_csv(
        JOB_PROFILES_FILE,
        low_memory=False,
    )

    validate_columns(
        matching,
        {"resume_id", "job_id", "baseline_score", "semantic_similarity"},
        "Baseline",
    )
    validate_columns(
        resumes,
        {
            "resume_id",
            "role",
            "seniority",
            "years_experience",
            "education",
        },
        "Processed resumes",
    )
    validate_columns(
        jobs,
        {
            "job_id",
            "normalized_title",
            "experience_level",
            "years_experience_numeric",
            "education_level",
            "preferred_qualifications",
        },
        "Processed jobs",
    )
    validate_columns(
        resume_profiles,
        {"resume_id", "candidate_skills"},
        "Resume skill profiles",
    )
    validate_columns(
        job_profiles,
        {"job_id", "required_skills"},
        "Job skill profiles",
    )

    # Normalize join keys.
    for dataframe, column in [
        (matching, "resume_id"),
        (matching, "job_id"),
        (resumes, "resume_id"),
        (jobs, "job_id"),
        (resume_profiles, "resume_id"),
        (job_profiles, "job_id"),
    ]:
        dataframe[column] = normalize_id(dataframe[column])

    # Ensure lookup keys are unique.
    for dataframe, column, name in [
        (resumes, "resume_id", "processed resumes"),
        (jobs, "job_id", "processed jobs"),
        (resume_profiles, "resume_id", "resume skill profiles"),
        (job_profiles, "job_id", "job skill profiles"),
    ]:
        if dataframe[column].duplicated().any():
            raise ValueError(f"Duplicate IDs found in {name}: {column}")

    # Keep only the columns needed for non-skill features.
    resumes_small = resumes[
        [
            "resume_id",
            "role",
            "seniority",
            "years_experience",
            "education",
        ]
    ].copy()

    jobs_small = jobs[
        [
            "job_id",
            "normalized_title",
            "experience_level",
            "years_experience_numeric",
            "education_level",
            "preferred_qualifications",
        ]
    ].copy()

    # Drop columns that this script recalculates.
    computed_columns = {
        "required_skill_count",
        "matched_required_skill_count",
        "missing_required_skill_count",
        "required_skill_match_ratio",
        "preferred_skill_count",
        "matched_preferred_skill_count",
        "preferred_skill_match_ratio",
        "skill_coverage",
        "role_compatibility",
        "experience_compatibility",
        "seniority_compatibility",
        "education_compatibility",
    }

    matching = matching.drop(
        columns=[
            column
            for column in computed_columns
            if column in matching.columns
        ],
        errors="ignore",
    )

    logger.info("Merging resume information...")

    features = matching.merge(
        resumes_small,
        on="resume_id",
        how="left",
        validate="many_to_one",
        indicator="_resume_merge",
    )

    resume_match_rate = (
        features["_resume_merge"] == "both"
    ).mean()

    logger.info(
        "Resume ID match rate: %.2f%%",
        resume_match_rate * 100,
    )

    features = features.drop(columns="_resume_merge")

    logger.info("Merging job information...")

    features = features.merge(
        jobs_small,
        on="job_id",
        how="left",
        validate="many_to_one",
        indicator="_job_merge",
    )

    job_match_rate = (
        features["_job_merge"] == "both"
    ).mean()

    logger.info(
        "Job ID match rate: %.2f%%",
        job_match_rate * 100,
    )

    if resume_match_rate < 0.99 or job_match_rate < 0.99:
        raise ValueError(
            "Processed dataset merge validation failed: "
            f"resume match rate={resume_match_rate:.2%}, "
            f"job match rate={job_match_rate:.2%}."
        )

    features = features.drop(columns="_job_merge")

    logger.info("Merging authoritative skill profiles...")

    features = features.merge(
        resume_profiles[["resume_id", "candidate_skills"]],
        on="resume_id",
        how="left",
        validate="many_to_one",
        indicator="_resume_profile_merge",
    )

    resume_profile_match_rate = (
        features["_resume_profile_merge"] == "both"
    ).mean()

    features = features.drop(columns="_resume_profile_merge")

    features = features.merge(
        job_profiles[["job_id", "required_skills"]],
        on="job_id",
        how="left",
        validate="many_to_one",
        indicator="_job_profile_merge",
    )

    job_profile_match_rate = (
        features["_job_profile_merge"] == "both"
    ).mean()

    features = features.drop(columns="_job_profile_merge")

    logger.info(
        "Resume profile match rate: %.2f%%",
        resume_profile_match_rate * 100,
    )
    logger.info(
        "Job profile match rate: %.2f%%",
        job_profile_match_rate * 100,
    )

    if resume_profile_match_rate < 0.99 or job_profile_match_rate < 0.99:
        raise ValueError(
            "Skill profile merge validation failed: "
            f"resume profile match rate={resume_profile_match_rate:.2%}, "
            f"job profile match rate={job_profile_match_rate:.2%}."
        )

    if features["candidate_skills"].isna().any():
        raise ValueError("Some matched resumes have missing skill profiles.")

    if features["required_skills"].isna().any():
        raise ValueError("Some matched jobs have missing skill profiles.")

    logger.info("Parsing skill fields...")
    parse_started = time.perf_counter()
    features = parse_skill_fields(features)
    logger.info(
        "Skill parsing completed in %.1fs",
        time.perf_counter() - parse_started,
    )

    logger.info("Building the production ranking feature vector...")
    feature_started = time.perf_counter()
    feature_rows = build_model_features_with_progress(features)
    model_features = pd.DataFrame(
        feature_rows.tolist(),
        index=features.index,
    )
    logger.info(
        "Feature vector construction completed in %.1fs",
        time.perf_counter() - feature_started,
    )

    if tuple(model_features.columns) != MODEL_FEATURE_NAMES:
        raise ValueError("Feature builder output differs from the model contract")

    source_baseline = pd.to_numeric(features["baseline_score"], errors="coerce")
    if source_baseline.isna().any() or not np.isfinite(source_baseline.to_numpy()).all():
        raise ValueError("Baseline scores contain missing or non-finite values")
    expected_baseline = model_features["baseline_score"].to_numpy(dtype=float)
    # The upstream baseline CSV rounds skill_coverage to four decimals while
    # baseline_score was calculated from its unrounded value.
    if not np.isclose(
        source_baseline.to_numpy(),
        expected_baseline,
        rtol=1e-6,
        atol=5e-5,
    ).all():
        raise ValueError(
            "Baseline scores do not match the shared training/inference formula; "
            "reconcile the source skill coverage before building features."
        )

    if model_features.isna().any().any() or not np.isfinite(
        model_features.to_numpy(dtype=float)
    ).all():
        raise ValueError("Ranking feature values contain missing or non-finite data")

    output = pd.concat(
        [features[["job_id", "resume_id"]], model_features],
        axis=1,
    )
    if output.columns.duplicated().any():
        raise ValueError("Output contains duplicate column names")

    logger.info("Rows in output: %s", len(output))
    logger.info("Unique jobs: %s", output["job_id"].nunique())
    logger.info("Unique resumes: %s", output["resume_id"].nunique())

    logger.info(
        "Pairs with required skill matches: %s",
        int((output["matched_required_skill_count"] > 0).sum()),
    )
    logger.info(
        "Mean required skill match ratio: %.4f",
        output["required_skill_match_ratio"].mean(),
    )
    logger.info(
        "Pairs with preferred skill matches: %s",
        int((output["matched_preferred_skill_count"] > 0).sum()),
    )

    # Preserve the existing output until all checks above pass.
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output.to_csv(output_file, index=False)

    logger.info("Saved features to %s", output_file)


if __name__ == "__main__":
    main()
