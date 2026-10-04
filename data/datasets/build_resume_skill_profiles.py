import pandas as pd
from pathlib import Path

INPUT_FILE = Path(
    "data/processed/resume_skills_normalized.csv"
)

OUTPUT_FILE = Path(
    "data/processed/resume_skill_profiles.csv"
)

print("Loading normalized resume skills...")

df = pd.read_csv(INPUT_FILE)

print(f"Skill records loaded: {len(df)}")


# Group skills by resume
profiles = (
    df.groupby("resume_id")["canonical_skill"]
    .apply(
        lambda skills: sorted(
            set(
                skill
                for skill in skills
                if isinstance(skill, str)
                and skill.strip()
            )
        )
    )
    .reset_index()
)


profiles["candidate_skills"] = profiles[
    "canonical_skill"
].apply(
    lambda skills: "|".join(skills)
)

profiles["candidate_skill_count"] = profiles[
    "canonical_skill"
].apply(len)


profiles = profiles[
    [
        "resume_id",
        "candidate_skills",
        "candidate_skill_count"
    ]
]


# Save
OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

profiles.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 70)
print("RESUME SKILL PROFILES CREATED")
print("=" * 70)

print(
    f"Total resume profiles: "
    f"{len(profiles)}"
)

print(
    f"Average skills per resume: "
    f"{profiles['candidate_skill_count'].mean():.2f}"
)

print(
    f"Minimum skills per resume: "
    f"{profiles['candidate_skill_count'].min()}"
)

print(
    f"Maximum skills per resume: "
    f"{profiles['candidate_skill_count'].max()}"
)


print("\nSample resume skill profiles:")

print(
    profiles.head(10).to_string(
        index=False
    )
)


print(f"\nSaved to: {OUTPUT_FILE}")