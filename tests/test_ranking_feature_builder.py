import re

import pandas as pd
import pytest

from backend.app.services.skill_service import skill_service
from data.datasets import build_ranking_features as feature_builder


def legacy_extract_skills(text):
    """Reference the pre-optimization implementation for equivalence tests."""
    if not text or not text.strip():
        return []

    normalized_text = skill_service.normalize_skill(text)
    matched_skills = []
    for skill in skill_service.skills:
        pattern = rf"(?<!\w){re.escape(skill)}(?!\w)"
        if re.search(pattern, normalized_text, flags=re.IGNORECASE):
            matched_skills.append((skill, skill_service.skill_mapping[skill]))

    selected = []
    for skill, canonical in matched_skills:
        if any(
            skill != selected_skill and skill in selected_skill
            for selected_skill, _ in selected
        ):
            continue
        selected.append((skill, canonical))

    return sorted({canonical for _, canonical in selected})


@pytest.mark.parametrize(
    "text",
    [
        None,
        "",
        "   ",
        "Senior Python engineer using C++, C#, Node.js, and .NET.",
        "Machine learning and learning; malformed [[qualification text",
        "Preferred: Python, SQL; malformed list fragment ['Rust', ???",
    ],
)
def test_precompiled_skill_matching_matches_legacy(text):
    assert skill_service.extract_skills(text) == legacy_extract_skills(text)


def test_skill_fields_are_parsed_once_per_resume_job_and_qualification(monkeypatch):
    frame = pd.DataFrame(
        [
            {
                "resume_id": "R1",
                "candidate_skills": "['Python', 'SQL']",
                "job_id": "J1",
                "required_skills": "['Python']",
                "preferred_qualifications": "Python experience",
            },
            {
                "resume_id": "R1",
                "candidate_skills": "['Python', 'SQL']",
                "job_id": "J2",
                "required_skills": "['C++']",
                "preferred_qualifications": None,
            },
            {
                "resume_id": "R2",
                "candidate_skills": "['Java']",
                "job_id": "J1",
                "required_skills": "['Python']",
                "preferred_qualifications": "Python experience",
            },
        ]
    )

    parse_calls = []
    original_parse = feature_builder.parse_list

    def parse_spy(value):
        parse_calls.append(value)
        return original_parse(value)

    preferred_calls = []

    def preferred_spy(value):
        preferred_calls.append(value)
        return [value] if value else []

    monkeypatch.setattr(feature_builder, "parse_list", parse_spy)
    monkeypatch.setattr(
        feature_builder.skill_service,
        "extract_skills",
        preferred_spy,
    )

    result = feature_builder.parse_skill_fields(frame)

    assert len(parse_calls) == 4  # two resumes plus two jobs
    assert len(preferred_calls) == 2  # one blank and one unique text
    assert result.loc[0, "resume_skills_list"] == ["python", "sql"]
    assert result.loc[1, "resume_skills_list"] == ["python", "sql"]
    assert result.loc[0, "required_skills_list"] == ["python"]
    assert result.loc[2, "required_skills_list"] == ["python"]
    assert result.loc[0, "preferred_skills_list"] == ["Python experience"]
    assert result.loc[2, "preferred_skills_list"] == ["Python experience"]
    assert result.loc[1, "preferred_skills_list"] == []


def test_parse_list_handles_empty_and_malformed_skill_lists():
    assert feature_builder.parse_list(None) == []
    assert feature_builder.parse_list(" ") == []
    assert feature_builder.parse_list("['Python', malformed") == [
        "malformed",
        "python",
    ]
