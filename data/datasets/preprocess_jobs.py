import pandas as pd
from pathlib import Path
import ast


INPUT_FILE = Path("data/raw/job_postings_raw.csv")
OUTPUT_FILE = Path("data/processed/job_postings_processed.csv")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


print("Loading job posting dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Original rows: {len(df)}")
print(f"Original columns: {len(df.columns)}")


# --------------------------------------------------
# 1. Select relevant columns
# --------------------------------------------------

selected_columns = [
    "title",
    "normalized_title",
    "company_name",
    "industry",
    "function",
    "occupational_category",
    "employment_type",
    "work_model",
    "experience_level",
    "job_level_normalized",
    "years_experience_numeric",
    "education_level",
    "skills_required",
    "minimum_qualifications",
    "preferred_qualifications",
    "responsibilities",
    "certifications",
    "job_description",
    "salary_min",
    "salary_max",
    "salary_currency",
    "city",
    "country",
    "benefits",
    "visa_sponsorship_available",
    "relocation_assistance",
    "date_posted",
    "closing_date"
]

df = df[selected_columns].copy()


# --------------------------------------------------
# 2. Remove completely empty job descriptions
# --------------------------------------------------

df["job_description"] = df["job_description"].fillna("").astype(str)

df = df[df["job_description"].str.strip() != ""]


# --------------------------------------------------
# 3. Clean text columns
# --------------------------------------------------

text_columns = [
    "title",
    "normalized_title",
    "company_name",
    "industry",
    "function",
    "occupational_category",
    "employment_type",
    "work_model",
    "experience_level",
    "job_level_normalized",
    "education_level",
    "minimum_qualifications",
    "preferred_qualifications",
    "certifications",
    "job_description",
    "city",
    "country",
    "benefits"
]

for column in text_columns:
    df[column] = df[column].fillna("").astype(str).str.strip()


# --------------------------------------------------
# 4. Parse skills_required
# --------------------------------------------------

def parse_list(value):

    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):
            return [
                str(item).strip()
                for item in parsed
                if str(item).strip()
            ]

    except (ValueError, SyntaxError):
        pass

    return []


df["skills_required"] = df["skills_required"].apply(parse_list)


# --------------------------------------------------
# 5. Create combined job text
# --------------------------------------------------

def create_job_text(row):

    skills = ", ".join(row["skills_required"])

    return (
        f"Job Title: {row['title']}. "
        f"Industry: {row['industry']}. "
        f"Function: {row['function']}. "
        f"Experience Level: {row['experience_level']}. "
        f"Required Experience: {row['years_experience_numeric']}. "
        f"Education: {row['education_level']}. "
        f"Required Skills: {skills}. "
        f"Minimum Qualifications: {row['minimum_qualifications']}. "
        f"Preferred Qualifications: {row['preferred_qualifications']}. "
        f"Responsibilities: {row['responsibilities']}. "
        f"Certifications: {row['certifications']}. "
        f"Description: {row['job_description']}"
    )


df["job_text"] = df.apply(create_job_text, axis=1)


# --------------------------------------------------
# 6. Add unique job ID
# --------------------------------------------------

df.insert(
    0,
    "job_id",
    ["JOB_{:06d}".format(i) for i in range(len(df))]
)


# --------------------------------------------------
# 7. Save processed dataset
# --------------------------------------------------

df.to_csv(OUTPUT_FILE, index=False)


print("\nProcessing completed.")

print(f"Processed rows: {len(df)}")
print(f"Processed columns: {len(df.columns)}")
print(f"Saved to: {OUTPUT_FILE}")

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst processed job:")
print(df.iloc[0].to_dict())