import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

INPUT_FILE = Path("data/processed/job_postings_processed.csv")
OUTPUT_FILE = Path("data/processed/job_embeddings.npy")

BATCH_SIZE = 64


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

print("=" * 70)
print("JOB EMBEDDING GENERATION")
print("=" * 70)

print("\nLoading job dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Jobs loaded: {len(df):,}")


# ---------------------------------------------------------
# Validate required column
# ---------------------------------------------------------

if "job_text" not in df.columns:
    raise ValueError(
        "Column 'job_text' was not found in the dataset."
    )


texts = df["job_text"].fillna("").tolist()


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

print("\nLoading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Model loaded successfully.")

print(
    f"Embedding dimension: "
    f"{model.get_embedding_dimension()}"
)


# ---------------------------------------------------------
# Generate embeddings
# ---------------------------------------------------------

print("\nGenerating job embeddings...")

embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True
)


# ---------------------------------------------------------
# Convert to NumPy array
# ---------------------------------------------------------

embeddings = np.asarray(
    embeddings,
    dtype=np.float32
)


# ---------------------------------------------------------
# Save embeddings
# ---------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

np.save(
    OUTPUT_FILE,
    embeddings
)


# ---------------------------------------------------------
# Verification
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("JOB EMBEDDING GENERATION COMPLETE")
print("=" * 70)

print(f"Number of embeddings: {len(embeddings):,}")

print(f"Embedding shape: {embeddings.shape}")

print(f"Data type: {embeddings.dtype}")

print(f"Saved to: {OUTPUT_FILE}")

print("\nFirst embedding preview:")

print(embeddings[0][:10])