from datasets import load_dataset
from collections import Counter

DATASET_NAME = "michaelozon/candidate-matching-synthetic"

print("Loading dataset...")
dataset = load_dataset(DATASET_NAME)

resumes = dataset["resumes"]

print("\n========== BASIC INFORMATION ==========")
print("Total resumes:", len(resumes))
print("Columns:", resumes.column_names)

print("\n========== ROLES ==========")
roles = Counter(resumes["role"])
for role, count in roles.most_common(15):
    print(f"{role}: {count}")

print("\n========== SENIORITY ==========")
seniority = Counter(resumes["seniority"])
for level, count in seniority.most_common():
    print(f"{level}: {count}")

print("\n========== INDUSTRIES ==========")
industries = Counter(resumes["industry"])
for industry, count in industries.most_common(15):
    print(f"{industry}: {count}")

print("\n========== EDUCATION ==========")
education = Counter(resumes["education"])
for degree, count in education.most_common():
    print(f"{degree}: {count}")

print("\n========== YEARS OF EXPERIENCE ==========")
years = resumes["years_experience"]
print("Minimum:", min(years))
print("Maximum:", max(years))
print("Average:", round(sum(years) / len(years), 2))

print("\n========== UNIQUE SKILLS ==========")
all_skills = []

for skills in resumes["skills"]:
    all_skills.extend(skills)

skill_counts = Counter(all_skills)

print("Total skill occurrences:", len(all_skills))
print("Unique skills:", len(skill_counts))

for skill, count in skill_counts.most_common(30):
    print(f"{skill}: {count}")