import pandas as pd
import re
from pathlib import Path

INPUT_FILE = Path("data/raw/resumes_raw.csv")
OUTPUT_FILE = Path("data/processed/resumes_processed.csv")


def parse_list(value):
    """Parse NumPy-style list strings such as:
    ['OOP' 'Databases' 'Git' 'Docker' 'Python']
    """

    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    # Extract text inside single quotes
    items = re.findall(r"'([^']*)'", value)

    if items:
        return [item.strip() for item in items if item.strip()]

    return []


def clean_text(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def create_resume_text(row):

    skills = ", ".join(row["skills"])
    experience = " ".join(row["experience_bullets"])

    return (
        f"Role: {row['role']}. "
        f"Seniority: {row['seniority']}. "
        f"Industry: {row['industry']}. "
        f"Education: {row['education']}. "
        f"Skills: {skills}. "
        f"Summary: {row['summary']}. "
        f"Experience: {experience}"
    )


print("Loading resume dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Original rows: {len(df)}")

# Remove duplicate resumes
df = df.drop_duplicates(subset="resume_id")

# Clean text columns
for column in [
    "role",
    "seniority",
    "industry",
    "education",
    "summary"
]:
    df[column] = df[column].apply(clean_text)

# Parse skills
df["skills"] = df["skills"].apply(parse_list)

# Parse experience bullets
df["experience_bullets"] = df["experience_bullets"].apply(parse_list)

# Create combined resume text
df["resume_text"] = df.apply(create_resume_text, axis=1)

# Save processed dataset
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

print("\nProcessing complete!")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Saved to: {OUTPUT_FILE}")

print("\nFirst resume skills:")
print(df["skills"].iloc[0])

print("\nFirst resume skill count:")
print(len(df["skills"].iloc[0]))

print("\nFirst resume text:")
print(df["resume_text"].iloc[0])