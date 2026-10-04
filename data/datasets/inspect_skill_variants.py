import pandas as pd
from pathlib import Path
from collections import Counter


INPUT_FILE = Path("data/processed/job_skills_normalized.csv")


print("Loading normalized skills...")

df = pd.read_csv(INPUT_FILE)

print(f"Total skill mappings: {len(df)}")
print(f"Unique canonical skills: {df['canonical_skill'].nunique()}")


# ---------------------------------------------------------
# 1. Show skills containing common variant indicators
# ---------------------------------------------------------

keywords = [
    "java",
    "python",
    "javascript",
    "react",
    "angular",
    "vue",
    "node",
    "sql",
    "aws",
    "azure",
    "google cloud",
    "docker",
    "kubernetes",
    "machine learning",
    "deep learning",
    "data analysis",
    "project management",
    "communication",
    "leadership",
    "excel",
    "power bi",
]


print("\n" + "=" * 70)
print("POTENTIAL SKILL VARIANTS")
print("=" * 70)

canonical_skills = sorted(
    df["canonical_skill"].dropna().unique()
)


for keyword in keywords:

    matches = [
        skill
        for skill in canonical_skills
        if keyword in skill
    ]

    if matches:

        print(f"\n[{keyword.upper()}]")

        for skill in matches[:50]:
            print(f"  - {skill}")


# ---------------------------------------------------------
# 2. Find skills that differ only by punctuation
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("PUNCTUATION VARIANTS")
print("=" * 70)


def simplify(skill):
    return (
        skill
        .lower()
        .replace("-", "")
        .replace(".", "")
        .replace("/", "")
        .replace(" ", "")
        .replace("_", "")
    )


groups = {}

for skill in canonical_skills:

    key = simplify(skill)

    groups.setdefault(key, []).append(skill)


punctuation_groups = [
    skills
    for skills in groups.values()
    if len(skills) > 1
]


for group in sorted(
    punctuation_groups,
    key=lambda x: (-len(x), x)
)[:100]:

    print("  |  ".join(group))


# ---------------------------------------------------------
# 3. Save all canonical skills for inspection
# ---------------------------------------------------------

OUTPUT_FILE = Path(
    "data/processed/canonical_skill_vocabulary.txt"
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as file:

    for skill in canonical_skills:
        file.write(skill + "\n")


print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)

print(
    f"\nSaved complete canonical vocabulary to:"
    f"\n{OUTPUT_FILE}"
)