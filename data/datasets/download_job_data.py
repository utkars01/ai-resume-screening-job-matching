from datasets import load_dataset
from pathlib import Path

DATASET_NAME = "NextGig-Rocks/global-job-postings-multi-ats"

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading job posting dataset...")

dataset = load_dataset(DATASET_NAME, split="train")

print(f"Total job postings: {len(dataset)}")

df = dataset.to_pandas()

output_file = OUTPUT_DIR / "job_postings_raw.csv"

df.to_csv(output_file, index=False)

print(f"\nSaved dataset to: {output_file}")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

print("\nColumns:")
print(df.columns.tolist())