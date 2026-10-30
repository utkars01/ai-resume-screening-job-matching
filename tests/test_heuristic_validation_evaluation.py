import numpy as np
import pandas as pd
import pytest

from data.datasets.evaluate_heuristic_validation import (
    evaluate_heuristic_validation,
    parse_heuristic_labels,
)


def test_heuristic_labels_reject_missing_invalid_and_single_class_data():
    with pytest.raises(ValueError, match="missing or invalid"):
        parse_heuristic_labels(pd.DataFrame({"relevance_label": [0, None]}))
    with pytest.raises(ValueError, match="0, 1, or 2"):
        parse_heuristic_labels(pd.DataFrame({"relevance_label": [0, 3]}))
    with pytest.raises(ValueError, match="only one class"):
        parse_heuristic_labels(pd.DataFrame({"relevance_label": [0, 0]}))


def test_automated_evaluation_reports_heuristic_source_and_limits(monkeypatch, tmp_path):
    from data.datasets import evaluate_heuristic_validation as evaluator

    class FakeBooster:
        feature_names = ["baseline_score"]

        def load_model(self, _path):
            pass

        def num_features(self):
            return 1

        def predict(self, matrix):
            return np.asarray(matrix.get_data().todense()).ravel()

    monkeypatch.setattr(evaluator.xgb, "Booster", FakeBooster)
    validation = tmp_path / "validation.csv"
    model = tmp_path / "model.json"
    features = tmp_path / "features.txt"
    pd.DataFrame(
        [
            ("job-1", "resume-a", 2, 0.1),
            ("job-1", "resume-b", 0, 0.9),
            ("job-2", "resume-c", 1, 0.2),
            ("job-2", "resume-d", 0, 0.5),
        ],
        columns=["job_id", "resume_id", "relevance_label", "baseline_score"],
    ).to_csv(validation, index=False)
    model.write_text("fixture")
    features.write_text("baseline_score\n")

    report, summary = evaluate_heuristic_validation(validation, model, features)

    assert summary["pairs"] == 4
    assert summary["jobs"] == 2
    assert summary["varying_jobs"] == 2
    assert summary["label_counts"] == {2: 1, 0: 2, 1: 1}
    assert "automatically generated labels; not ground truth" in report
    assert "build_relevance_labels.py" in report
    assert "percentile >= 0.90" in report
    assert "do not validate model quality" in report
