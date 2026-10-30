from backend.app.services.applicant_metadata import extract_applicant_metadata


def test_extracts_labeled_role_years_and_degree_variants():
    result = extract_applicant_metadata(
        """Avery Example
Role: Senior Product Manager
Over 7 years of professional experience
Bachelor's degree in Computer Science
"""
    )

    assert result == {
        "role": "Senior Product Manager",
        "years_experience": 7,
        "seniority": "senior",
        "education": "Bachelor's degree in Computer Science",
    }


def test_extracts_unlabeled_header_and_short_experience_units():
    result = extract_applicant_metadata(
        """Jordan Lee
Cloud Infrastructure Engineer
Experience: 4 yrs
M.S. Computer Science
"""
    )

    assert result["role"] == "Cloud Infrastructure Engineer"
    assert result["years_experience"] == 4
    assert result["seniority"] == "mid"
    assert result["education"] == "M.S. Computer Science"


def test_preserves_unknown_values_instead_of_inventing_them():
    result = extract_applicant_metadata(
        """Taylor Example
Skills: Python, SQL, communication
Completed projects in healthcare
"""
    )

    assert result == {
        "role": None,
        "years_experience": None,
        "seniority": None,
        "education": None,
    }


def test_extracts_experience_from_a_resume_summary():
    result = extract_applicant_metadata(
        """Data Analyst
8+ yrs in business intelligence
MBA, Business Analytics
"""
    )

    assert result["years_experience"] == 8
    assert result["seniority"] == "senior"
    assert result["education"] == "MBA, Business Analytics"


def test_derives_years_from_employment_dates_and_merges_overlap():
    result = extract_applicant_metadata(
        """Jordan Example
Professional Experience
Software Engineer | Acme | Jan 2020 - Jan 2023
Consultant | Example Co | Jan 2022 - Jan 2024
Education
Bachelor of Engineering
"""
    )

    assert result["years_experience"] == 4
    assert result["seniority"] == "mid"


def test_does_not_treat_a_manager_job_title_as_seniority():
    result = extract_applicant_metadata(
        """Role: Product Manager
Bachelor of Arts
"""
    )

    assert result["role"] == "Product Manager"
    assert result["seniority"] is None
