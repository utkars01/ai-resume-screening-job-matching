import time
import pandas as pd
from sentence_transformers import SentenceTransformer


INPUT_FILE = "data/processed/job_postings_processed.csv"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TEST_SIZE = 1000
BATCH_SIZE = 256


print("=" * 70)
print("JOB EMBEDDING SPEED TEST")
print("=" * 70)

print("\nLoading jobs...")

df = pd.read_csv(INPUT_FILE, nrows=TEST_SIZE)

texts = df["job_text"].fillna("").tolist()

print(f"Jobs loaded for test: {len(texts):,}")


print("\nLoading model...")

model = SentenceTransformer(MODEL_NAME)

print("Model loaded.")


print(
    f"\nTesting batch size: {BATCH_SIZE}"
)

start_time = time.time()

embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True
)

elapsed = time.time() - start_time


print("\n" + "=" * 70)
print("SPEED TEST COMPLETE")
print("=" * 70)

print(f"Jobs processed: {len(texts):,}")

print(f"Time taken: {elapsed:.2f} seconds")

print(
    f"Speed: "
    f"{len(texts) / elapsed:.2f} jobs/second"
)

estimated_seconds = (
    112816 / (len(texts) / elapsed)
)

estimated_minutes = estimated_seconds / 60

print(
    f"\nEstimated full-dataset time: "
    f"{estimated_minutes:.1f} minutes"
)

print(
    f"Embedding shape: {embeddings.shape}"
)