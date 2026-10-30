from types import SimpleNamespace

from scripts.backfill_applicant_metadata import plan_backfill


def test_backfill_plan_only_fills_recoverable_null_fields_without_mutating():
    applicant = SimpleNamespace(
        id=7,
        extracted_text=(
            "Senior Data Engineer\n"
            "Professional Experience\n"
            "Data Engineer | 2018 - 2022\n"
            "Education\nBachelor of Science"
        ),
        role="Manually reviewed role",
        years_experience=None,
        seniority=None,
        education=None,
    )
    no_text = SimpleNamespace(
        id=8, extracted_text="", role=None, years_experience=None,
        seniority=None, education=None,
    )

    plan = plan_backfill([applicant, no_text])

    assert set(plan) == {7}
    assert plan[7]["years_experience"] == 4
    assert plan[7]["seniority"] == "senior"
    assert plan[7]["education"].startswith("Bachelor of Science")
    assert "role" not in plan[7]
    assert applicant.role == "Manually reviewed role"
    assert applicant.years_experience is None
    assert applicant.seniority is None
    assert applicant.education is None
    assert no_text.role is None
