# ------------------------------------------------------------------------------
# Stage 1: Build the React + Vite Frontend
# ------------------------------------------------------------------------------
FROM node:22-alpine AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# ------------------------------------------------------------------------------
# Stage 2: Production Python API, Machine Learning & Static Web Runtime
# ------------------------------------------------------------------------------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000 \
    HF_HOME=/app/hf-cache \
    EMBEDDING_MODEL_REVISION=1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
    EMBEDDING_LOCAL_FILES_ONLY=true \
    UPLOAD_STORAGE_DIR=/app/data/uploads

WORKDIR /app

# Install system dependencies (libgomp1 is required for XGBoost OpenMP runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (using PyTorch CPU build to minimize image size)
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Pre-cache pinned Sentence Transformers embedding model snapshot for 100% offline startup
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41')"

# Copy application source code and ML artifacts
COPY backend /app/backend
COPY models /app/models
COPY data/processed/skill_master.csv /app/data/processed/skill_master.csv
COPY scripts /app/scripts

# Copy production frontend build from Stage 1
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

# Set up entrypoint script and storage directories
COPY scripts/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh && mkdir -p /app/data/uploads

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
