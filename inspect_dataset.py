from datasets import load_dataset

DATASET_NAME = "michaelozon/candidate-matching-synthetic"

print("Loading dataset...")
dataset = load_dataset(DATASET_NAME)

print("\nDataset structure:")
print(dataset)

for split in dataset:
    print(f"\n--- {split} ---")
    print("Rows:", len(dataset[split]))
    print("Columns:")
    print(dataset[split].column_names)

    print("\nFirst record:")
    print(dataset[split][0])