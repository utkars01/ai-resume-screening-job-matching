import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# CONFIGURATION
# ============================================================

COVERAGE_FILE = Path("data/processed/skill_coverage_top_candidates.csv")
RESUME_EMBEDDINGS_FILE = Path("data/processed/resume_embeddings.npy")
JOB_EMBEDDINGS_FILE = Path("data/processed/relevant_job_embeddings.npy")
JOB_IDS_FILE = Path("data/processed/relevant_job_ids.csv")
RESUME_FILE = Path("data/processed/resumes_processed.csv")
JOB_FILE = Path("data/processed/job_postings_processed.csv")

OUTPUT_FILE = Path("data/processed/baseline_matching_results.csv")

SEMANTIC_WEIGHT = 0.70
SKILL_WEIGHT = 0.30

# ============================================================
# START
# ============================================================

print("=" * 70)
print("BASELINE AI MATCHING ENGINE")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load embeddings
# ------------------------------------------------------------

print("\nLoading resume embeddings...")
resume_embeddings = np.load(RESUME_EMBEDDINGS_FILE)

print(f"Resume embeddings: {resume_embeddings.shape}")

print("\nLoading job embeddings...")
job_embeddings = np.load(JOB_EMBEDDINGS_FILE)

print(f"Job embeddings: {job_embeddings.shape}")

# ------------------------------------------------------------
# 2. Load ID mappings
# ------------------------------------------------------------

print("\nLoading resume data...")
resume_df = pd.read_csv(RESUME_FILE)

print(f"Resumes: {len(resume_df)}")

print("\nLoading job ID mapping...")
job_ids_df = pd.read_csv(JOB_IDS_FILE)

print(f"Relevant jobs: {len(job_ids_df)}")

# Create ID -> embedding index mappings

resume_id_to_index = {
    resume_id: index
    for index, resume_id in enumerate(resume_df["resume_id"])
}

job_id_to_index = {
    job_id: index
    for index, job_id in enumerate(job_ids_df["job_id"])
}

# ------------------------------------------------------------
# 3. Load skill coverage candidates
# ------------------------------------------------------------

print("\nLoading skill coverage candidates...")

coverage_df = pd.read_csv(COVERAGE_FILE)

print(f"Coverage records: {len(coverage_df):,}")

# ------------------------------------------------------------
# 4. Validate IDs
# ------------------------------------------------------------

print("\nValidating IDs...")

coverage_df = coverage_df[
    coverage_df["resume_id"].isin(resume_id_to_index)
    & coverage_df["job_id"].isin(job_id_to_index)
].copy()

print(f"Valid coverage records: {len(coverage_df):,}")

# ------------------------------------------------------------
# 5. Convert IDs to embedding indexes
# ------------------------------------------------------------

coverage_df["resume_index"] = coverage_df["resume_id"].map(
    resume_id_to_index
)

coverage_df["job_index"] = coverage_df["job_id"].map(
    job_id_to_index
)

# ------------------------------------------------------------
# 6. Calculate semantic similarity
# ------------------------------------------------------------

print("\nCalculating semantic similarity...")

resume_indices = coverage_df["resume_index"].to_numpy()
job_indices = coverage_df["job_index"].to_numpy()

resume_vectors = resume_embeddings[resume_indices]
job_vectors = job_embeddings[job_indices]

# Embeddings were normalized during generation,
# so cosine similarity is simply the dot product.

semantic_scores = np.sum(
    resume_vectors * job_vectors,
    axis=1
)

coverage_df["semantic_similarity"] = semantic_scores

print(
    f"Semantic similarity range: "
    f"{semantic_scores.min():.4f} - {semantic_scores.max():.4f}"
)

# ------------------------------------------------------------
# 7. Skill coverage
# ------------------------------------------------------------

coverage_df["skill_coverage"] = coverage_df["skill_coverage"].astype(float)

# ------------------------------------------------------------
# 8. Hybrid baseline score
# ------------------------------------------------------------

coverage_df["baseline_score"] = (
    SEMANTIC_WEIGHT * coverage_df["semantic_similarity"]
    + SKILL_WEIGHT * coverage_df["skill_coverage"]
)

# ------------------------------------------------------------
# 9. Rank candidates for each job
# ------------------------------------------------------------

print("\nRanking candidates...")

coverage_df = coverage_df.sort_values(
    ["job_id", "baseline_score"],
    ascending=[True, False]
)

coverage_df["rank"] = (
    coverage_df.groupby("job_id").cumcount() + 1
)

# ------------------------------------------------------------
# 10. Select top candidates
# ------------------------------------------------------------

top_candidates = coverage_df[
    coverage_df["rank"] <= 10
].copy()

# ------------------------------------------------------------
# 11. Save results
# ------------------------------------------------------------

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

top_candidates.to_csv(
    OUTPUT_FILE,
    index=False
)

# ------------------------------------------------------------
# 12. Display summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BASELINE MATCHING COMPLETE")
print("=" * 70)

print(f"Total candidate-job pairs evaluated: {len(coverage_df):,}")
print(f"Top candidates saved: {len(top_candidates):,}")
print(f"Jobs represented: {top_candidates['job_id'].nunique():,}")

print("\nScore statistics:")
print(
    top_candidates[
        [
            "semantic_similarity",
            "skill_coverage",
            "baseline_score"
        ]
    ].describe()
)

print("\nTop 10 candidates for first job:")

first_job = top_candidates["job_id"].iloc[0]

print(
    top_candidates[
        top_candidates["job_id"] == first_job
    ][
        [
            "job_id",
            "resume_id",
            "semantic_similarity",
            "skill_coverage",
            "baseline_score",
            "rank"
        ]
    ].to_string(index=False)
)

print("\nResults saved to:")
print(OUTPUT_FILE)