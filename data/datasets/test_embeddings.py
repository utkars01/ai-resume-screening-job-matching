from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Model loaded successfully.\n")


resume_text = """
Software engineer with experience in Python, machine learning,
SQL, REST APIs, Docker and backend development.
"""

job_text = """
We are looking for a backend developer who can build APIs,
work with Python and databases, and deploy applications
using container technologies.
"""

unrelated_job = """
We are looking for a digital marketing specialist with experience
in SEO, Google Ads, content marketing and social media campaigns.
"""


print("Generating embeddings...")

resume_embedding = model.encode(
    resume_text,
    normalize_embeddings=True
)

job_embedding = model.encode(
    job_text,
    normalize_embeddings=True
)

unrelated_embedding = model.encode(
    unrelated_job,
    normalize_embeddings=True
)


similar_job_score = cosine_similarity(
    [resume_embedding],
    [job_embedding]
)[0][0]

unrelated_job_score = cosine_similarity(
    [resume_embedding],
    [unrelated_embedding]
)[0][0]


print("\n" + "=" * 60)
print("SEMANTIC SIMILARITY TEST")
print("=" * 60)

print(f"\nResume ↔ Backend Job:     {similar_job_score:.4f}")

print(f"Resume ↔ Marketing Job:   {unrelated_job_score:.4f}")

print("\nEmbedding shape:")
print(resume_embedding.shape)

print("\nTest complete.")