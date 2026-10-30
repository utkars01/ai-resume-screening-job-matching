
import ast
import re
from pathlib import Path

import pandas as pd


# ============================================================
# 1. Paths and configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "job_postings_raw.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "job_postings_processed.csv"

MAX_EXPERIENCE_YEARS = 60


# ============================================================
# 2. Number words
# ============================================================

NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
}

NUMBER = (
    r"(?:\d+(?:\.\d+)?|"
    + "|".join(sorted(NUMBER_WORDS, key=len, reverse=True))
    + r")"
)


# ============================================================
# 3. Experience extraction patterns
# ============================================================

RANGE_PATTERN = re.compile(
    rf"\b(?P<minimum>{NUMBER})\s*[-–—]\s*"
    rf"(?P<maximum>{NUMBER})\s*"
    rf"(?P<unit>years?|months?)\b",
    re.IGNORECASE,
)

PLUS_PATTERN = re.compile(
    rf"\b(?P<minimum>{NUMBER})\s*\+\s*"
    rf"(?P<unit>years?|months?)\b",
    re.IGNORECASE,
)

SINGLE_PATTERN = re.compile(
    rf"\b(?P<minimum>{NUMBER})\s*"
    rf"(?P<unit>years?|months?)\b",
    re.IGNORECASE,
)

AGE_PATTERN = re.compile(
    r"\b(?:"
    r"(?:at least\s+)?\d+\s*\+?\s*years?\s+old"
    r"|aged\s+(?:\d+|[a-z]+)"
    r"|age requirement"
    r")\b",
    re.IGNORECASE,
)

NO_EXPERIENCE_PATTERN = re.compile(
    r"\bno\s+"
    r"(?:(?:prior|previous|professional|work)\s+)?"
    r"experience\s+(?:is\s+)?"
    r"(?:required|needed|necessary)\b"
    r"|\bno experience necessary\b"
    r"|\bno experience needed\b"
    r"|\bwithout (?:any )?experience\b",
    re.IGNORECASE,
)


# ============================================================
# 4. Utility functions
# ============================================================

def is_missing(value):
    """Safely identify missing scalar values."""
    if value is None:
        return True

    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def clean_text(value):
    """Convert a value to clean text without turning NaN into 'nan'."""
    if is_missing(value):
        return ""

    text = str(value).strip()

    if text.lower() in {"nan", "none", "null", "n/a"}:
        return ""

    return re.sub(r"\s+", " ", text)


def parse_experience_number(value):
    """Convert a supported numeric or written number into a float."""
    value = clean_text(value).lower()

    if value in NUMBER_WORDS:
        return float(NUMBER_WORDS[value])

    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def parse_experience_value(value, unit):
    """Convert an experience value into years."""
    number = parse_experience_number(value)

    if number is None or number < 0:
        return None

    if unit.lower().startswith("month"):
        number /= 12.0

    if number > MAX_EXPERIENCE_YEARS:
        return None

    return round(number, 6)


def parse_list_or_text(value):
    """
    Parse a list stored as text, such as:
    ["Python", "SQL"]

    If the value is not a list representation, return the cleaned text.
    """
    text = clean_text(value)

    if not text:
        return ""

    try:
        parsed = ast.literal_eval(text)

        if isinstance(parsed, (list, tuple, set)):
            return ", ".join(
                clean_text(item)
                for item in parsed
                if clean_text(item)
            )
    except (ValueError, SyntaxError, TypeError):
        pass

# ============================================================
# 5. Extract minimum experience requirement
# ============================================================



# ============================================================
# Education requirement extraction
# ============================================================

EDUCATION_PATTERNS = [
    (
        r"\b(?:ph\.?d\.?|doctorate|doctoral degree)\b",
        "Doctorate",
    ),
    (
        r"\b(?:master(?:'s)?(?:\s+degree)?|mba|m\.?s\.?)\b",
        "Master",
    ),
    (
        r"\b(?:bachelor(?:'s)?(?:\s+degree)?|b\.?s\.?|b\.?a\.?|bsn)\b",
        "Bachelor",
    ),
    (
        r"\b(?:associate(?:'s)?(?:\s+degree)?|"
        r"associate degree|a\.a\.|a\.s\.)\b",
        "Associate",
    ),
    (
        r"\b(?:high school diploma|high school graduate|high school|ged)\b",
        "High School",
    ),
]

EDUCATION_PREFERENCE_PATTERN = re.compile(
    r"\b(?:preferred|preferably|nice to have|desired|a plus|"
    r"would be an advantage)\b",
    re.IGNORECASE,
)

EDUCATION_ALTERNATIVE_PATTERN = re.compile(
    r"\bor\b",
    re.IGNORECASE,
)


def extract_required_education(text):
    """Extract an unambiguous minimum education level."""
    if not isinstance(text, str) or not text.strip():
        return None

    try:
        parsed = ast.literal_eval(text)
        bullets = (
            [item for item in parsed if isinstance(item, str)]
            if isinstance(parsed, (list, tuple))
            else [text]
        )
    except (ValueError, SyntaxError):
        bullets = [text]


    # Reject role-specific lists containing different degree requirements.
    combined_text = " ".join(
        bullet.strip()
        for bullet in bullets
        if isinstance(bullet, str)
    )


    role_specific_pattern = re.compile(
        r"\b(?:lecturer|instructor|assistant professor|"
        r"associate professor|professor)\s*(?::|[-–—])"
        r"|\b(?:BSN|MSN|PhD|DNP)\s+for\s+"
        r"(?:lecturer|instructor|assistant professor|"
        r"associate professor|professor)\b",
        re.IGNORECASE,
    )


    if role_specific_pattern.search(combined_text):
        return None

    found_levels = set()

    for bullet in bullets:
        bullet = bullet.strip()

        # Enrollment does not establish a completed degree requirement.
        if re.search(
            r"\b(?:enrollment|enrolled|currently pursuing|"
            r"pursuing|working toward|working towards|"
            r"candidate for)\b",
            bullet,
            re.IGNORECASE,
        ):
            continue



                # Reject slash-separated lists of different degree credentials.
        # These may represent alternatives at different education levels.
        if re.search(
            r"\b(?:Ph\.?\s*D\.?|Pharm\.?\s*D\.?|M\.?\s*S\.?|"
            r"M\.?\s*D\.?|D\.?\s*N\.?\s*P\.?|"
            r"M\.?\s*B\.?\s*A\.?|M\.?\s*P\.?\s*H\.?)"
            r"\s*(?:/|,)\s*"
            r"(?:Ph\.?\s*D\.?|Pharm\.?\s*D\.?|M\.?\s*S\.?|"
            r"M\.?\s*D\.?|D\.?\s*N\.?\s*P\.?|"
            r"M\.?\s*B\.?\s*A\.?|M\.?\s*P\.?\s*H\.?)",
            bullet,
            re.IGNORECASE,
        ):
            continue

        # Reject comma-separated lists containing multiple degree credentials.
        if re.search(
            r"\bPh\.?\s*D\.?\b.*,\s*"
            r"(?:Pharm\.?\s*D\.?|D\.?\s*N\.?\s*P\.?|M\.?\s*D\.?)\b",
            bullet,
            re.IGNORECASE,
        ):
            continue


        # Skip qualifications that explicitly list multiple
        # degree pathways or allow experience as a substitute.
        if re.search(
            r"\b(?:or equivalent experience|equivalent experience)\b",
            bullet,
            re.IGNORECASE,
        ):
            continue

        if re.search(
            r"\b(?:BSc\s*/\s*MSc\s*/\s*PhD|"
            r"MD\s*/\s*PhD|PhD\s+or\s+M\.?D\.?/?D\.?O\.?|"
            r"PhD\s+or\s+PharmD|"
            r"MD\s+or\s+MD\s*/\s*PhD)\b",
            bullet,
            re.IGNORECASE,
        ):
            continue


        if not bullet or EDUCATION_PREFERENCE_PATTERN.search(bullet):
            continue

        matches = {
            level
            for pattern, level in EDUCATION_PATTERNS
            if re.search(pattern, bullet, re.IGNORECASE)
        }

        if len(matches) != 1:
            continue

        level = matches.pop()

        # Ignore ambiguous alternatives between degree levels.
        if re.search(
            r"\bor\s+(?:a\s+|an\s+)?"
            r"(?:master(?:'s)?|bachelor(?:'s)?|associate(?:'s)?|"
            r"doctorate|doctoral|ph\.?d\.?|mba|bsn)\b",
            bullet,
            re.IGNORECASE,
        ):
            continue
	        # Reject alternative doctoral/professional degree pathways.
        if re.search(
            r"\bph\.?\s*d\.?\s+or\s+"
            r"(?:m\.?\s*d\.?|d\.?\s*o\.?)",
            bullet,
            re.IGNORECASE,
        ):
            continue

        found_levels.add(level)

    # Do not guess when different required education levels are listed.
    if len(found_levels) != 1:
        return None

    return found_levels.pop()


# ============================================================
# 6. Regression tests
# ============================================================

REGRESSION_CASES = {
    "0-2 years relevant experience": 0.0,
    "1-2 years experience as a medical assistant": 1.0,
    "4+ years software engineering experience": 4.0,
    "3 years sales experience": 3.0,
    "4+ years implementation in merchant payment processing": 4.0,
    "5 years of experience in Python": 5.0,
    "Minimum five years maintenance experience": 5.0,
    "Two years nursing experience": 2.0,
    "1 year RN experience": 1.0,
    "At least 16 years old": None,
    "18+ years old": None,
    "No experience required": 0.0,
    "No prior experience required": 0.0,
    "No previous experience needed": 0.0,
    "Flexibility to work varied shifts": None,
    "Ability to lift up to 50 lbs": None,
    "": None,
    "No experience necessary": 0.0,
    "No experience needed": 0.0,
    "0–3 years of clinical experience": 0.0,
    "0–2 years relevant experience": 0.0,
    "0–1 years legal experience": 0.0,
    "Licensing requirement: 8+ years experience or Master's with 6+ years or PhD with 3+ years | 0–2 years supervisory experience": None,
    "At least 16 years old": None,
}

def run_regression_tests():
    print("Validating experience extraction rules...\n")

    for example, expected in REGRESSION_CASES.items():
        actual = extract_requirement(example)

        if actual != expected:
            raise AssertionError(
                f"FAILED: {example!r}; "
                f"expected {expected!r}, got {actual!r}"
            )

        print(f"PASS | {example!r} -> {actual}")

    print(
        f"\nAll {len(REGRESSION_CASES)} "
        "experience extraction checks passed."
    )


# ============================================================
# 7. Job preprocessing
# ============================================================

def preprocess_jobs():
    print("\nLoading job posting dataset...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    original_rows = len(df)

    print(f"Raw rows: {original_rows:,}")

    # Keep all original columns to avoid accidentally removing fields
    # used by downstream processing and matching.
    df = df.copy()

    # Fill missing education levels using explicit minimum qualifications.
    # Preserve every existing non-null education value.
    if "education_level" not in df.columns:
        df["education_level"] = pd.NA

    education_missing_before = int(df["education_level"].isna().sum())
    education_recovered = 0

    if "minimum_qualifications" in df.columns:
        for index in df.index[df["education_level"].isna()]:
            qualifications = clean_text(
                df.at[index, "minimum_qualifications"]
            )

            extracted = extract_required_education(qualifications)

            if extracted is not None:
                df.at[index, "education_level"] = extracted
                education_recovered += 1

    print(
        f"Education missing before extraction: "
        f"{education_missing_before:,}"
    )
    print(f"Education values recovered: {education_recovered:,}")
    print(
        f"Education missing after extraction: "
        f"{int(df['education_level'].isna().sum()):,}"
    )


    # Normalize commonly used text fields when they exist.
    text_columns = [
        "title",
        "normalized_title",
        "company_name",
        "industry",
        "job_description",
        "minimum_qualifications",
        "responsibilities",
        "experience_level",
        "job_level_normalized",
    ]

    for column in text_columns:
        if column in df.columns:
            df[column] = df[column].map(clean_text)

    # Parse list-formatted skill values without discarding other fields.
    if "skills_required" in df.columns:
        df["skills_required"] = df["skills_required"].map(
            parse_list_or_text
        )
    elif "skills" in df.columns:
        df["skills_required"] = df["skills"].map(
            parse_list_or_text
        )

    # Start with the existing numeric experience values, when available.
    if "years_experience_numeric" in df.columns:
        experience = pd.to_numeric(
            df["years_experience_numeric"],
            errors="coerce",
        )

        # Reject impossible values rather than propagating them.
        experience = experience.where(
            experience.between(0, MAX_EXPERIENCE_YEARS)
        )
    else:
        experience = pd.Series(
            float("nan"),
            index=df.index,
            dtype="float64",
        )

    missing_before = int(experience.isna().sum())

    # Extract experience from qualifications first. Use the description
    # only when the qualifications do not provide an explicit value.
    extracted_values = []

    for index, row in df.iterrows():
        existing_value = experience.loc[index]

        if pd.notna(existing_value):
            extracted_values.append(existing_value)
            continue

        qualifications = clean_text(
            row.get("minimum_qualifications", "")
        )

        description = clean_text(
            row.get("job_description", "")
        )


        extracted = extract_requirement(qualifications)

        # Do not infer minimum experience from general job descriptions.
        # Descriptions may contain unrelated numbers or experience statements.
        extracted_values.append(extracted)


    df["years_experience_numeric"] = pd.Series(
        extracted_values,
        index=df.index,
        dtype="float64",
    )

    recovered = missing_before - int(
        df["years_experience_numeric"].isna().sum()
    )

    # Generate job IDs only if they are absent.
    if "job_id" not in df.columns:
        df.insert(
            0,
            "job_id",
            [f"job_{i:06d}" for i in range(1, len(df) + 1)],
        )

    # Build a combined searchable text field.
    job_text_columns = [
        "title",
        "normalized_title",
        "industry",
        "job_description",
        "minimum_qualifications",
        "responsibilities",
        "skills_required",
    ]

    available_columns = [
        column
        for column in job_text_columns
        if column in df.columns
    ]

    if available_columns:
        df["job_text"] = df[available_columns].apply(
            lambda row: " ".join(
                clean_text(value)
                for value in row
                if clean_text(value)
            ),
            axis=1,
        )
    else:
        df["job_text"] = ""

    # Validate basic data integrity before writing.
    if len(df) != original_rows:
        raise AssertionError(
            f"Row count changed unexpectedly: "
            f"{original_rows:,} -> {len(df):,}"
        )

    if df["job_id"].isna().any():
        raise AssertionError("Some job IDs are missing.")

    if not df["job_id"].is_unique:
        raise AssertionError(
            "Duplicate job IDs detected. Output was not saved."
        )

    numeric_experience = pd.to_numeric(
        df["years_experience_numeric"],
        errors="coerce",
    )

    invalid_experience = numeric_experience.notna() & (
        (numeric_experience < 0)
        | (numeric_experience > MAX_EXPERIENCE_YEARS)
    )

    if invalid_experience.any():
        raise AssertionError(
            "Invalid experience values detected. Output was not saved."
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(OUTPUT_FILE, index=False)

    # ========================================================
    # 8. Summary
    # ========================================================

    print("\nPreprocessing completed.")
    print(f"Rows processed: {len(df):,}")
    print(f"Missing before extraction: {missing_before:,}")
    print(f"Newly recovered: {recovered:,}")
    print(
        "Missing after extraction: "
        f"{int(df['years_experience_numeric'].isna().sum()):,}"
    )
    print(
        "Experience values present: "
        f"{int(df['years_experience_numeric'].notna().sum()):,}"
    )
    print(
        "Experience values above 60 years: "
        f"{int((numeric_experience > MAX_EXPERIENCE_YEARS).sum()):,}"
    )
    print(f"Saved processed data to: {OUTPUT_FILE}")


# ============================================================
# 9. Main
# ============================================================

if __name__ == "__main__":
    run_regression_tests()
    preprocess_jobs()
