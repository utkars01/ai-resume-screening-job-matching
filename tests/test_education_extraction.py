import pytest

from data.datasets.preprocess_jobs import extract_required_education


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (["Ph.D. in Chemistry"], "Doctorate"),
        (["Doctoral degree in Clinical Psychology"], "Doctorate"),
        (["Doctorate degree (MD or DO)"], "Doctorate"),
        (["Degree in Engineering (BSc/MSc/PhD)"], None),
        (["PhD or PharmD"], None),
        (["Ph.D. or M.D./D.O. in microbiology"], None),
        (["PhD or equivalent experience in ML/AI"], None),
        (["Bachelor degree or Master degree"], None),
        (["Bachelor degree preferred"], None),
        (["Associate degree required"], "Associate"),
        (["High School Diploma or equivalent"], "High School"),
        (["PhD or PsyD in Psychology"], "Doctorate"),
        (["Ph.D. or Psy.D. in clinical psychology"], "Doctorate"),
        (["PhD/PharmD/MD/MSN"], None),
        (["PhD, PharmD, DNP, or MD"], None),
        (["MD/DO/OD/PhD/PharmD preferred"], None),
        (
            ["BSN for Lecturer", "MSN or equivalent for Instructor",
             "PhD/DNP or equivalent for Professor"],
            None,
        ),
        (
            ["Lecturer: BSN", "Instructor: MSN or equivalent",
             "Professor: PhD/DNP or equivalent"],
            None,
        ),
        (["Enrollment in a PhD or PsyD program"], None),
    ],
)
def test_extract_required_education(text, expected):
    assert extract_required_education(str(text)) == expected


@pytest.mark.parametrize("text", [None, "", "   ", 123])
def test_extract_required_education_handles_invalid_input(text):
    assert extract_required_education(text) is None
