from datasets import load_dataset
import pandas as pd
from pathlib import Path

DATASET_NAME = "michaelozon/candidate-matching-synthetic"

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset...")
dataset = load_dataset(DATASET_NAME)

resumes = dataset["resumes"]

print(f"Total resumes: {len(resumes)}")

df = resumes.to_pandas()

output_file = OUTPUT_DIR / "resumes_raw.csv"

df.to_csv(output_file, index=False)

print(f"\nSaved dataset to: {output_file}")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")