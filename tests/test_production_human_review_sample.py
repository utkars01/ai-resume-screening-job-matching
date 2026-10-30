import numpy as np
import pandas as pd
import pytest

from data.datasets.create_production_human_review_sample import select_review_pairs
from data.datasets.create_production_human_review_sample import (
    create_review_sample,
    write_review_instructions,
)


def test_sample_is_diversified_deterministic_and_contains_no_scores():
    frame = pd.DataFrame(
        [
            (job, f"resume-{index}", float(9 - index), float(index))
            for job in ("job-a", "job-b")
            for index in range(10)
        ],
        columns=["job_id", "resume_id", "baseline_score", "unused"],
    )
    model_scores = np.tile(np.arange(10, dtype=float), 2)

    first = select_review_pairs(
        frame, model_scores, jobs_to_sample=2, candidates_per_job=4, seed=7
    )
    second = select_review_pairs(
        frame, model_scores, jobs_to_sample=2, candidates_per_job=4, seed=7
    )

    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == ["job_id", "resume_id"]
    assert len(first) == 8
    assert first.groupby("job_id").size().eq(4).all()
    assert not first.duplicated(["job_id", "resume_id"]).any()
    assert set(first.loc[first.job_id == "job-a", "resume_id"]) == {
        "resume-0", "resume-1", "resume-8", "resume-9"
    }


def test_review_instructions_use_create_only_mode(tmp_path):
    guide = tmp_path / "review_instructions.txt"
    write_review_instructions(guide)
    text = guide.read_text(encoding="utf-8")
    assert "0 = poor fit" in text
    assert "1 = partial fit" in text
    assert "2 = strong fit" in text
    with pytest.raises(FileExistsError):
        write_review_instructions(guide)


def test_sample_builder_refuses_to_overwrite_existing_output(tmp_path):
    output = tmp_path / "review.csv"
    output.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_review_sample(
            tmp_path / "missing_validation.csv",
            tmp_path / "missing_resumes.csv",
            tmp_path / "missing_jobs.csv",
            tmp_path / "missing_model.json",
            tmp_path / "missing_features.txt",
            output,
        )

    assert output.read_text(encoding="utf-8") == "keep"
