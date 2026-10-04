import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# BASELINE MATCHING EVALUATION
# ============================================================

print("=" * 70)
print("BASELINE MATCHING EVALUATION")
print("=" * 70)


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "processed"

RESULTS_FILE = PROCESSED_DIR / "baseline_matching_results.csv"
REPORT_FILE = PROCESSED_DIR / "baseline_evaluation.txt"


# ------------------------------------------------------------
# Load baseline results
# ------------------------------------------------------------

print("\nLoading baseline matching results...")

df = pd.read_csv(RESULTS_FILE)

print(f"Matching records: {len(df):,}")
print(f"Jobs represented: {df['job_id'].nunique():,}")
print(f"Resumes represented: {df['resume_id'].nunique():,}")


# ------------------------------------------------------------
# Basic validation
# ------------------------------------------------------------

required_columns = [
    "job_id",
    "resume_id",
    "semantic_similarity",
    "skill_coverage",
    "baseline_score",
    "rank"
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")


# ------------------------------------------------------------
# Ranking coverage
# ------------------------------------------------------------

jobs = df["job_id"].nunique()

rank_counts = df.groupby("job_id")["rank"].max()

jobs_with_10 = (rank_counts >= 10).sum()
jobs_with_5 = (rank_counts >= 5).sum()
jobs_with_1 = (rank_counts >= 1).sum()

coverage_10 = jobs_with_10 / jobs * 100
coverage_5 = jobs_with_5 / jobs * 100
coverage_1 = jobs_with_1 / jobs * 100


# ------------------------------------------------------------
# Top-K evaluation
# ------------------------------------------------------------

def top_k_metrics(data, k):

    top_k = data[data["rank"] <= k]

    grouped = top_k.groupby("job_id")

    metrics = {
        "mean_semantic_similarity":
            grouped["semantic_similarity"].mean().mean(),

        "mean_skill_coverage":
            grouped["skill_coverage"].mean().mean(),

        "mean_baseline_score":
            grouped["baseline_score"].mean().mean(),

        "mean_max_skill_coverage":
            grouped["skill_coverage"].max().mean(),

        "mean_max_semantic_similarity":
            grouped["semantic_similarity"].max().mean(),

        "jobs_with_skill_coverage_50":
            (grouped["skill_coverage"].max() >= 0.50).mean() * 100,

        "jobs_with_skill_coverage_75":
            (grouped["skill_coverage"].max() >= 0.75).mean() * 100,

        "jobs_with_full_skill_match":
            (grouped["skill_coverage"].max() >= 1.00).mean() * 100,
    }

    return metrics


metrics_1 = top_k_metrics(df, 1)
metrics_5 = top_k_metrics(df, 5)
metrics_10 = top_k_metrics(df, 10)


# ------------------------------------------------------------
# Correlation analysis
# ------------------------------------------------------------

pearson_corr = df[
    ["semantic_similarity", "skill_coverage"]
].corr().loc["semantic_similarity", "skill_coverage"]

spearman_corr = df[
    ["semantic_similarity", "skill_coverage"]
].corr(method="spearman").loc[
    "semantic_similarity", "skill_coverage"
]


# ------------------------------------------------------------
# Score statistics
# ------------------------------------------------------------

score_stats = df[
    [
        "semantic_similarity",
        "skill_coverage",
        "baseline_score"
    ]
].describe()


# ------------------------------------------------------------
# Top candidate statistics
# ------------------------------------------------------------

top1 = df[df["rank"] == 1].copy()

top1_mean_score = top1["baseline_score"].mean()
top1_mean_semantic = top1["semantic_similarity"].mean()
top1_mean_skill = top1["skill_coverage"].mean()


# ------------------------------------------------------------
# Skill coverage distribution
# ------------------------------------------------------------

skill_distribution = {
    ">= 25%": (df["skill_coverage"] >= 0.25).mean() * 100,
    ">= 50%": (df["skill_coverage"] >= 0.50).mean() * 100,
    ">= 75%": (df["skill_coverage"] >= 0.75).mean() * 100,
    "100%": (df["skill_coverage"] >= 1.00).mean() * 100,
}


# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("RANKING COVERAGE")
print("=" * 70)

print(f"Jobs with Rank 1 candidate : {jobs_with_1:,} ({coverage_1:.2f}%)")
print(f"Jobs with Top 5 candidates : {jobs_with_5:,} ({coverage_5:.2f}%)")
print(f"Jobs with Top 10 candidates: {jobs_with_10:,} ({coverage_10:.2f}%)")


print("\n" + "=" * 70)
print("TOP-K PERFORMANCE DIAGNOSTICS")
print("=" * 70)

for k, metrics in [
    (1, metrics_1),
    (5, metrics_5),
    (10, metrics_10)
]:

    print(f"\nTop-{k}")

    print(
        f"  Mean semantic similarity : "
        f"{metrics['mean_semantic_similarity']:.4f}"
    )

    print(
        f"  Mean skill coverage     : "
        f"{metrics['mean_skill_coverage']:.4f}"
    )

    print(
        f"  Mean baseline score     : "
        f"{metrics['mean_baseline_score']:.4f}"
    )

    print(
        f"  Mean max skill coverage : "
        f"{metrics['mean_max_skill_coverage']:.4f}"
    )

    print(
        f"  Jobs with >=50% skill match: "
        f"{metrics['jobs_with_skill_coverage_50']:.2f}%"
    )

    print(
        f"  Jobs with >=75% skill match: "
        f"{metrics['jobs_with_skill_coverage_75']:.2f}%"
    )

    print(
        f"  Jobs with 100% skill match: "
        f"{metrics['jobs_with_full_skill_match']:.2f}%"
    )


print("\n" + "=" * 70)
print("CORRELATION ANALYSIS")
print("=" * 70)

print(
    f"Pearson correlation "
    f"(semantic vs skill): {pearson_corr:.4f}"
)

print(
    f"Spearman correlation "
    f"(semantic vs skill): {spearman_corr:.4f}"
)


print("\n" + "=" * 70)
print("TOP-1 CANDIDATE")
print("=" * 70)

print(f"Mean semantic similarity: {top1_mean_semantic:.4f}")
print(f"Mean skill coverage     : {top1_mean_skill:.4f}")
print(f"Mean baseline score     : {top1_mean_score:.4f}")


print("\n" + "=" * 70)
print("SKILL COVERAGE DISTRIBUTION")
print("=" * 70)

for threshold, percentage in skill_distribution.items():
    print(f"{threshold:>8} : {percentage:.2f}%")


# ------------------------------------------------------------
# Save report
# ------------------------------------------------------------

with open(REPORT_FILE, "w", encoding="utf-8") as f:

    f.write("BASELINE MATCHING EVALUATION\n")
    f.write("=" * 70 + "\n\n")

    f.write("DATASET\n")
    f.write("-" * 70 + "\n")
    f.write(f"Matching records: {len(df):,}\n")
    f.write(f"Jobs represented: {jobs:,}\n")
    f.write(f"Resumes represented: {df['resume_id'].nunique():,}\n\n")

    f.write("RANKING COVERAGE\n")
    f.write("-" * 70 + "\n")
    f.write(
        f"Jobs with Rank 1 candidate: "
        f"{jobs_with_1:,} ({coverage_1:.2f}%)\n"
    )
    f.write(
        f"Jobs with Top 5 candidates: "
        f"{jobs_with_5:,} ({coverage_5:.2f}%)\n"
    )
    f.write(
        f"Jobs with Top 10 candidates: "
        f"{jobs_with_10:,} ({coverage_10:.2f}%)\n\n"
    )

    f.write("TOP-K DIAGNOSTICS\n")
    f.write("-" * 70 + "\n")

    for k, metrics in [
        (1, metrics_1),
        (5, metrics_5),
        (10, metrics_10)
    ]:

        f.write(f"\nTop-{k}\n")

        f.write(
            f"Mean semantic similarity: "
            f"{metrics['mean_semantic_similarity']:.4f}\n"
        )

        f.write(
            f"Mean skill coverage: "
            f"{metrics['mean_skill_coverage']:.4f}\n"
        )

        f.write(
            f"Mean baseline score: "
            f"{metrics['mean_baseline_score']:.4f}\n"
        )

        f.write(
            f"Mean max skill coverage: "
            f"{metrics['mean_max_skill_coverage']:.4f}\n"
        )

        f.write(
            f"Jobs with >=50% skill match: "
            f"{metrics['jobs_with_skill_coverage_50']:.2f}%\n"
        )

        f.write(
            f"Jobs with >=75% skill match: "
            f"{metrics['jobs_with_skill_coverage_75']:.2f}%\n"
        )

        f.write(
            f"Jobs with 100% skill match: "
            f"{metrics['jobs_with_full_skill_match']:.2f}%\n"
        )

    f.write("\n\nCORRELATION\n")
    f.write("-" * 70 + "\n")
    f.write(
        f"Pearson correlation: {pearson_corr:.4f}\n"
    )
    f.write(
        f"Spearman correlation: {spearman_corr:.4f}\n"
    )

    f.write("\n\nIMPORTANT NOTE\n")
    f.write("-" * 70 + "\n")
    f.write(
        "These are diagnostic baseline metrics, not true ranking "
        "performance metrics.\n"
    )
    f.write(
        "Formal Precision@K, Recall@K, MRR and NDCG require "
        "independent relevance labels.\n"
    )
    f.write(
        "The current project will create a weakly supervised "
        "evaluation/training dataset before the final ML ranking model.\n"
    )


print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print(f"\nReport saved to:")
print(REPORT_FILE)