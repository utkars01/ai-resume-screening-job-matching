import numpy as np
import pandas as pd
import pytest

from data.datasets.evaluate_human_review import (
    evaluate_review_sample,
    metrics_for_ranking,
    parse_review_labels,
)


def test_perfect_ranking_metrics_for_highest_relevance_first():
    metrics = metrics_for_ranking(np.array([2, 1, 0]))

    assert metrics["ndcg@10"] == pytest.approx(1.0)
    assert metrics["mrr"] == 1.0
    assert metrics["hit@1"] == 1.0
    assert metrics["hit@5"] == 1.0
    assert metrics["hit@10"] == 1.0


def test_review_labels_require_complete_valid_integer_values():
    with pytest.raises(ValueError, match="unlabeled"):
        parse_review_labels(pd.DataFrame({"reviewer_label": [0, ""]}))
    with pytest.raises(ValueError, match="0, 1, or 2"):
        parse_review_labels(pd.DataFrame({"reviewer_label": [0, 3]}))


def test_evaluation_joins_pairs_scores_both_rankers_and_reports_coverage(
    monkeypatch, tmp_path
):
    from data.datasets import evaluate_human_review as evaluator

    class FakeBooster:
        feature_names = ["feature"]

        def load_model(self, _path):
            pass

        def num_features(self):
            return 1

        def predict(self, matrix):
            return np.asarray(matrix.get_data().todense()).ravel()

    monkeypatch.setattr(evaluator.xgb, "Booster", FakeBooster)
    review_path = tmp_path / "review.csv"
    validation_path = tmp_path / "validation.csv"
    model_path = tmp_path / "model.json"
    features_path = tmp_path / "features.txt"
    pd.DataFrame(
        [
            ("job-1", "resume-a", 2),
            ("job-1", "resume-b", 0),
            ("job-2", "resume-c", 1),
        ],
        columns=["job_id", "resume_id", "reviewer_label"],
    ).to_csv(review_path, index=False)
    pd.DataFrame(
        [
            ("job-1", "resume-a", 0.1, 3.0),
            ("job-1", "resume-b", 0.9, 1.0),
            ("job-2", "resume-c", 0.2, 2.0),
        ],
        columns=["job_id", "resume_id", "baseline_score", "feature"],
    ).to_csv(validation_path, index=False)
    model_path.write_text("fixture")
    features_path.write_text("feature\n")

    report, summary = evaluate_review_sample(
        review_path, validation_path, model_path, features_path
    )

    assert summary == {
        "jobs": 2,
        "pairs": 3,
        "coverage": 1.0,
        "label_counts": {2: 1, 0: 1, 1: 1},
    }
    assert "Candidate-pair coverage for these jobs: 100.0%" in report
    assert "LIMITATIONS" in report
