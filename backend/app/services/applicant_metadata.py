"""Conservative extraction of basic applicant metadata from resume text."""

import re
from datetime import date


_ROLE_LABEL = re.compile(
    r"^(?:current\s+)?(?:job\s+)?(?:role|title|position|headline)\s*[:\-]\s*(.+)$",
    re.IGNORECASE,
)
_ROLE_TERMS = re.compile(
    r"\b(?:engineer|developer|analyst|scientist|designer|architect|"
    r"consultant|manager|administrator|specialist|researcher|accountant|"
    r"technician|recruiter|programmer|product\s+owner|data\s+entry)\b",
    re.IGNORECASE,
)
_SENIORITY = (
    ("intern", re.compile(r"\b(?:intern|internship)\b", re.IGNORECASE)),
    ("junior", re.compile(r"\b(?:junior|jr\.?|entry[ -]?level)\b", re.IGNORECASE)),
    ("senior", re.compile(r"\b(?:senior|sr\.?)\b", re.IGNORECASE)),
    ("lead", re.compile(r"\b(?:lead|principal|staff)\b", re.IGNORECASE)),
)
_EXPERIENCE_PATTERNS = (
    re.compile(
        r"\b(?:over|more\s+than)\s+(\d{1,2})\s*\+?\s*(?:years?|yrs?)\s+"
        r"(?:of\s+)?(?:professional\s+)?experience\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(\d{1,2})\s*\+?\s*(?:years?|yrs?)\s+"
        r"(?:of\s+)?(?:professional\s+)?experience\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bexperience\s*[:\-]\s*(\d{1,2})\s*\+?\s*(?:years?|yrs?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(\d{1,2})\s*\+?\s*(?:years?|yrs?)\s+(?:in|as|working)\b",
        re.IGNORECASE,
    ),
)
_MONTH = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
_MONTH += r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|"
_MONTH += r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_DATE_RANGE = re.compile(
    rf"(?P<start>(?:{_MONTH}\s+)?(?:19|20)\d{{2}})\s*"
    rf"(?:-|\u2013|\u2014|\bto\b)\s*"
    rf"(?P<end>(?:(?:{_MONTH}\s+)?(?:19|20)\d{{2}})|present|current)",
    re.IGNORECASE,
)
_MONTH_NUMBER = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_EXPERIENCE_HEADINGS = re.compile(
    r"\b(?:professional\s+experience|work\s+experience|employment\s+history|"
    r"career\s+history|experience)\b",
    re.IGNORECASE,
)
_SECTION_HEADING = re.compile(
    r"^(?:education|academic|skills|certifications|projects|awards|references)\b",
    re.IGNORECASE,
)
_EDUCATION_TERMS = re.compile(
    r"\b(?:bachelor(?:'s|s)?|master(?:'s|s)?|b\.?\s?tech|m\.?\s?tech|"
    r"b\.?\s?e\.?|m\.?\s?e\.?|b\.?\s?sc\.?|m\.?\s?sc\.?|"
    r"b\.?\s?a\.?|m\.?\s?a\.?|b\.?\s?s\.?|m\.?\s?s\.?|"
    r"bca|mca|mba|ph\.?d\.?|doctorate|associate(?:'s)?\s+degree)\b",
    re.IGNORECASE,
)


def _clean_line(line: str) -> str:
    return re.sub(r"^\s*(?:[-*•▪]|\d+[.)])\s*", "", line).strip()


def _extract_role(lines: list[str]) -> str | None:
    # Explicit labels are the strongest evidence and do not require a fixed
    # vocabulary of role names.
    for line in lines[:40]:
        match = _ROLE_LABEL.match(_clean_line(line))
        if match:
            value = match.group(1).split(" | ")[0].strip(" -|,")
            if value:
                return value[:255]

    # Handle common resume headers such as "Senior Data Engineer" without
    # guessing from the skill or employment-description sections.
    for line in lines[:25]:
        candidate = _clean_line(line).split(" | ")[0].strip(" -|,")
        if (
            len(candidate) <= 120
            and _ROLE_TERMS.search(candidate)
            and not re.search(r"\b(?:experience|responsibilit|skills)\b", candidate, re.I)
        ):
            return candidate[:255]
    return None


def _extract_years_experience(text: str) -> int | None:
    for pattern in _EXPERIENCE_PATTERNS:
        match = pattern.search(text)
        if match:
            return int(match.group(1))

    lines = text.splitlines()
    in_experience = False
    intervals = []
    for line in lines:
        cleaned = _clean_line(line)
        if _EXPERIENCE_HEADINGS.search(cleaned):
            in_experience = True
            continue
        if in_experience and _SECTION_HEADING.match(cleaned):
            break
        if not in_experience:
            continue

        for match in _DATE_RANGE.finditer(cleaned):
            start = _date_to_month(match.group("start"))
            end = _date_to_month(match.group("end"))
            if start is not None and end is not None and end >= start:
                intervals.append((start, end))

    if not intervals:
        return None

    # Merge overlapping jobs so concurrent roles are not double counted.
    intervals.sort()
    total_months = 0
    range_start, range_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= range_end:
            range_end = max(range_end, end)
        else:
            total_months += range_end - range_start
            range_start, range_end = start, end
    total_months += range_end - range_start
    return min(total_months // 12, 60)


def _date_to_month(value: str) -> int | None:
    normalized = value.strip().lower()
    if normalized in {"present", "current"}:
        today = date.today()
        return today.year * 12 + today.month

    match = re.match(rf"(?:(?P<month>{_MONTH})\s+)?(?P<year>(?:19|20)\d{{2}})$", normalized, re.I)
    if not match:
        return None
    month_text = match.group("month")
    month = _MONTH_NUMBER[month_text[:3].lower()] if month_text else 1
    # Year-only resume dates are interpreted at the start of the listed year
    # so ranges such as 2020–2023 do not claim four full years.
    return int(match.group("year")) * 12 + month


def extract_applicant_metadata(text: str) -> dict:
    """Return only qualifications supported by explicit resume text."""
    lines = [_clean_line(line) for line in text.splitlines() if line.strip()]
    role = _extract_role(lines)
    years_experience = _extract_years_experience(text)

    education = next(
        (line[:255] for line in lines if _EDUCATION_TERMS.search(line)),
        None,
    )

    seniority = None
    for value, pattern in _SENIORITY:
        if pattern.search(role or ""):
            seniority = value
            break

    if seniority is None and years_experience is not None:
        if years_experience <= 2:
            seniority = "junior"
        elif years_experience <= 5:
            seniority = "mid"
        else:
            seniority = "senior"

    return {
        "role": role,
        "years_experience": years_experience,
        "seniority": seniority,
        "education": education,
    }
