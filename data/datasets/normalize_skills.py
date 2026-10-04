import pandas as pd
import ast
import re
import unicodedata
from pathlib import Path


INPUT_FILE = Path("data/processed/job_postings_processed.csv")
OUTPUT_FILE = Path("data/processed/job_skills_normalized.csv")


# High-confidence aliases only.
# We will expand this dictionary later after inspecting the data.
ALIASES = {
    "ms office": "microsoft office",
    "microsoft office suite": "microsoft office",
    "microsoft office package": "microsoft office",

    "reactjs": "react",
    "react.js": "react",
    "react js": "react",

    "nodejs": "node.js",
    "node js": "node.js",

    "vuejs": "vue.js",
    "vue js": "vue.js",

    "angularjs": "angular",
    "angular js": "angular",

    "nextjs": "next.js",
    "next js": "next.js",

    "typescript": "typescript",

    "postgres": "postgresql",
    "postgres sql": "postgresql",

    "mongodb database": "mongodb",

    "powerbi": "power bi",

    "scikit learn": "scikit-learn",
    "sklearn": "scikit-learn",

    "restful api": "rest api",
    "restful apis": "rest api",
    "rest apis": "rest api",

    "machine learning": "machine learning",
    "ml": "machine learning",

    "artificial intelligence": "artificial intelligence",
    "ai": "artificial intelligence",

    "natural language processing": "natural language processing",
    "nlp": "natural language processing",

    "continuous integration continuous deployment": "ci/cd",
    "continuous integration/continuous deployment": "ci/cd",

    "object oriented programming": "object-oriented programming",
    "object-oriented programming": "object-oriented programming",
    "oop": "object-oriented programming",

    "sql server": "microsoft sql server",
    "mssql": "microsoft sql server",

    "aws cloud": "aws",
    "amazon web services": "aws",

    "google cloud platform": "google cloud",
    "gcp cloud": "google cloud",

    "microsoft azure cloud": "azure",
    "azure cloud": "azure",
}


def normalize_skill(skill):
    """Basic text normalization."""

    if not isinstance(skill, str):
        return ""

    skill = unicodedata.normalize("NFKC", skill)

    # Normalize special hyphens
    skill = skill.replace("\u2011", "-")
    skill = skill.replace("\u2013", "-")
    skill = skill.replace("\u2014", "-")

    # Remove leading/trailing whitespace
    skill = skill.strip()

    # Collapse multiple spaces
    skill = re.sub(r"\s+", " ", skill)

    # Lowercase
    skill = skill.lower()

    return skill


def canonicalize_skill(skill):
    """Apply basic normalization followed by high-confidence aliases."""

    normalized = normalize_skill(skill)

    if not normalized:
        return ""

    return ALIASES.get(normalized, normalized)


def parse_skills(value):
    """Convert stored list string into a Python list."""

    if pd.isna(value):
        return []

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):
            return [
                str(skill).strip()
                for skill in parsed
                if str(skill).strip()
            ]

    except (ValueError, SyntaxError):
        return []

    return []


print("Loading job dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Jobs loaded: {len(df)}")


skill_records = []

for _, row in df.iterrows():

    skills = parse_skills(row["skills_required"])

    for skill in skills:

        normalized = normalize_skill(skill)
        canonical = canonicalize_skill(skill)

        if normalized:

            skill_records.append({
                "original_skill": skill,
                "normalized_skill": normalized,
                "canonical_skill": canonical
            })


skills_df = pd.DataFrame(skill_records)


# Remove duplicate mappings
skills_df = skills_df.drop_duplicates(
    subset=[
        "original_skill",
        "normalized_skill",
        "canonical_skill"
    ]
)


OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

skills_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nNormalization complete!")

print(f"Total skill records: {len(skill_records)}")
print(f"Unique original skills: {skills_df['original_skill'].nunique()}")
print(f"Unique normalized skills: {skills_df['normalized_skill'].nunique()}")
print(f"Unique canonical skills: {skills_df['canonical_skill'].nunique()}")


print("\nAlias examples:")

examples = [
    "MS Office",
    "Microsoft Office",
    "ReactJS",
    "React.js",
    "React JS",
    "NodeJS",
    "Node JS",
    "PowerBI",
    "Sklearn",
    "RESTful API",
    "ML",
    "NLP",
    "OOP",
    "AWS Cloud",
    "Amazon Web Services",
]


for example in examples:

    canonical = canonicalize_skill(example)

    print(
        f"{example:30} -> {canonical}"
    )


print(f"\nSaved to: {OUTPUT_FILE}")