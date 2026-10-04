import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

JOB_FILE = Path("data/processed/job_postings_processed.csv")

COVERAGE_FILE = Path(
    "data/processed/skill_coverage_top_candidates.csv"
)

OUTPUT_EMBEDDINGS = Path(
    "data/processed/relevant_job_embeddings.npy"
)

OUTPUT_IDS = Path(
    "data/processed/relevant_job_ids.csv"
)

BATCH_SIZE = 256


# ---------------------------------------------------------
# Load relevant job IDs
# ---------------------------------------------------------

print("=" * 70)
print("RELEVANT JOB EMBEDDING GENERATION")
print("=" * 70)

print("\nLoading skill coverage data...")

coverage = pd.read_csv(COVERAGE_FILE)

job_ids = (
    coverage["job_id"]
    .drop_duplicates()
    .tolist()
)

print(
    f"Relevant jobs identified: {len(job_ids):,}"
)


# ---------------------------------------------------------
# Load job dataset
# ---------------------------------------------------------

print("\nLoading job dataset...")

jobs = pd.read_csv(JOB_FILE)

print(
    f"Total jobs loaded: {len(jobs):,}"
)


# ---------------------------------------------------------
# Select relevant jobs
# ---------------------------------------------------------

print("\nSelecting relevant jobs...")

relevant_jobs = jobs[
    jobs["job_id"].isin(job_ids)
].copy()


# Preserve the exact order of job IDs
job_id_order = {
    job_id: index
    for index, job_id in enumerate(job_ids)
}

relevant_jobs["_order"] = (
    relevant_jobs["job_id"]
    .map(job_id_order)
)

relevant_jobs = relevant_jobs.sort_values(
    "_order"
).drop(
    columns="_order"
).reset_index(drop=True)


print(
    f"Relevant jobs loaded: "
    f"{len(relevant_jobs):,}"
)


# ---------------------------------------------------------
# Validate alignment
# ---------------------------------------------------------

if len(relevant_jobs) != len(job_ids):

    missing_ids = set(job_ids) - set(
        relevant_jobs["job_id"]
    )

    raise ValueError(
        f"Some job IDs were not found. "
        f"Missing: {len(missing_ids)}"
    )


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

print("\nGenerating relevant job embeddings...")

texts = (
    relevant_jobs["job_text"]
    .fillna("")
    .tolist()
)

embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True
)


# ---------------------------------------------------------
# Convert to float32
# ---------------------------------------------------------

embeddings = np.asarray(
    embeddings,
    dtype=np.float32
)


# ---------------------------------------------------------
# Save embeddings
# ---------------------------------------------------------

OUTPUT_EMBEDDINGS.parent.mkdir(
    parents=True,
    exist_ok=True
)

np.save(
    OUTPUT_EMBEDDINGS,
    embeddings
)


# Save matching job IDs
relevant_jobs[
    ["job_id"]
].to_csv(
    OUTPUT_IDS,
    index=False
)


# ---------------------------------------------------------
# Verification
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("RELEVANT JOB EMBEDDING GENERATION COMPLETE")
print("=" * 70)

print(
    f"Number of embeddings: "
    f"{len(embeddings):,}"
)

print(
    f"Embedding shape: "
    f"{embeddings.shape}"
)

print(
    f"Data type: "
    f"{embeddings.dtype}"
)

print(
    f"\nEmbeddings saved to:"
    f"\n{OUTPUT_EMBEDDINGS}"
)

print(
    f"\nJob IDs saved to:"
    f"\n{OUTPUT_IDS}"
)

print("\nFirst 5 job IDs:")

print(
    relevant_jobs["job_id"].head().to_string(
        index=False
    )
)

print("\nFirst embedding preview:")

print(
    embeddings[0][:10]
)