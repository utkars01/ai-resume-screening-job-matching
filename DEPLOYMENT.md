# Deployment Guide

## 1. Overview

This document describes how to prepare and deploy the AI Resume Screening & Job Matching application.

The application consists of:

- **Backend:** FastAPI
- **Database:** PostgreSQL
- **Frontend:** React and Vite
- **Ranking:** XGBoost
- **Semantic processing:** Sentence Transformers

The deployment process requires a compatible Python environment, Node.js for frontend builds, persistent storage, PostgreSQL, and the pinned embedding model.

> **Deployment status:** Local validation has been performed. Public deployment, hosting-provider compatibility, and production infrastructure have not yet been verified.

## 2. Runtime requirements

Use the following versions:

| Component | Requirement |
|---|---|
| Python | 3.12 |
| Node.js | 22 LTS (22.12 or newer) |
| Database | PostgreSQL |
| Frontend | Vite production build |
| Backend | FastAPI with Uvicorn |

Install the locked Python runtime dependencies:

```bash
python -m pip install --require-hashes -r requirements-runtime.lock
```

Install frontend dependencies and build the production frontend:

```bash
npm ci --prefix frontend
npm run build --prefix frontend
```

`requirements-runtime.in` defines direct runtime dependencies, while `requirements-runtime.lock` contains the complete hashed dependency resolution.

Development and testing dependencies are maintained separately in `requirements.txt`.

## 3. Build the release package

Build the frontend before creating the release package.

Create a new, empty output directory outside the repository workspace:

```bash
python scripts/build_release_candidate.py --output /absolute/path/to/new/release-candidate
```

The release builder:

- Includes backend application code and versioned migrations.
- Includes the production frontend build.
- Includes the production XGBoost model and feature definitions.
- Includes the verified runtime skill vocabulary.
- Includes the pinned local Sentence Transformers model snapshot.
- Generates a release manifest containing artifact hashes.
- Excludes raw datasets, processed datasets, applicant files, environment files, backups, tests, and experimental models.

The builder refuses to overwrite an existing output directory.

The production model and validated feature dataset must not be retrained or regenerated as part of deployment preparation without a separately reviewed process.

## 4. Runtime skill vocabulary

The application uses:

`data/processed/skill_master.csv`

This is the authoritative runtime skill vocabulary.

It is generated from the canonical skill mapping and skill-frequency inputs. The release builder verifies that the artifact is reproducible from its source inputs before packaging.

Raw and intermediate datasets are not required at runtime.

Do not regenerate or overwrite the validated skill vocabulary during deployment preparation.

Refer to `data/README.md` for dataset provenance and processing instructions.

## 5. Database preparation

Use a new, empty PostgreSQL database for a fresh deployment.

Configure the `DATABASE_URL` environment variable and run:

```bash
python scripts/migrate_database.py
```

The migration runner:

- Applies SQL migrations in numeric order.
- Records migration versions and SHA-256 checksums.
- Uses a PostgreSQL advisory lock.
- Applies migrations transactionally.
- Refuses to migrate a populated database without migration history.

**Important database precautions:**

- Do not point the migration runner at an existing application database without reviewing its schema and migration history.
- Do not run migrations against a configured local database as part of deployment preparation.
- Existing populated databases require a separately reviewed baseline-adoption procedure.
- Back up an existing database before any approved schema change.

## 6. Environment configuration

Configure the following variables through the hosting platform's secure environment-variable or secret-management facility.

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | PostgreSQL SQLAlchemy connection URL using `postgresql+psycopg` |
| `AUTH_SECRET` | Random secret containing at least 32 UTF-8 bytes |
| `DEBUG` | Set to `false` |
| `AUTH_COOKIE_SECURE` | Set to `true` when serving through HTTPS |
| `AUTH_TOKEN_TTL_MINUTES` | Session lifetime; default is 720 minutes |
| `UPLOAD_STORAGE_DIR` | Writable persistent directory for uploaded resumes |
| `EMBEDDING_MODEL_REVISION` | Pinned Sentence Transformers model revision |
| `EMBEDDING_LOCAL_FILES_ONLY` | Set to `true` |
| `HF_HOME` | Directory containing the packaged Hugging Face model cache |

The pinned embedding revision is:

```text
1110a243fdf4706b3f48f1d95db1a4f5529b4d41
```

Security requirements:

- Never commit `.env` files or credentials.
- Never expose server secrets through frontend `VITE_*` variables.
- Use HTTPS.
- Keep database access restricted to the application.
- Use persistent storage for uploaded resumes.
- Keep embedding-model access local-only in production.

The application rejects insecure production authentication configuration and online embedding lookup.

## 7. Backend startup

Run the API from the release root:

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1 --log-level info
```

The application loads the pinned embedding model during startup. Allow approximately 90 seconds for startup probes.

Health endpoints:

| Endpoint | Purpose |
|---|---|
| `/health` | Liveness check |
| `/ready` | Database and embedding-model readiness check |

Use one worker for the current application configuration. Login throttling is maintained in memory, and additional workers can multiply model memory consumption.

For multiple replicas, a shared rate limiter and shared durable upload storage are required.

## 8. Frontend and API routing

The production frontend is generated in:

```text
frontend/dist/
```

Serve the built frontend through a web server or hosting platform.

Configure the reverse proxy to forward same-origin API requests:

```text
/api/*  ->  FastAPI backend
```

The application does not enable CORS. The frontend and backend should therefore be exposed through a compatible same-origin routing configuration.

Ensure that the reverse proxy supports the application's upload limits:

- Maximum batch size: 50 files.
- Maximum request body: approximately 250 MiB plus multipart overhead.

If the hosting platform has lower request limits, reduce the application's upload limits accordingly.

## 9. Persistent storage and operational requirements

The deployment environment must provide:

- Persistent PostgreSQL storage.
- Persistent storage for uploaded resumes.
- Secure storage for environment variables.
- HTTPS and appropriate reverse-proxy configuration.
- Database backup and recovery procedures.
- Application logging with restricted access.
- Sufficient memory for the embedding model and ranking pipeline.

Do not use temporary or ephemeral directories for applicant uploads.

Application logs should not expose resume contents, authentication secrets, or database credentials.

## 10. Deployment validation checklist

Before making the application publicly accessible, verify:

- [ ] Runtime dependencies install successfully.
- [ ] Frontend production build completes.
- [ ] Release manifest is generated.
- [ ] Production model and feature definitions are present.
- [ ] Pinned embedding model loads successfully.
- [ ] A new, empty demo database migrates successfully.
- [ ] Authentication and secure cookies work over HTTPS.
- [ ] `/health` and `/ready` return expected results.
- [ ] Resume upload and parsing work.
- [ ] Ranking requests complete successfully.
- [ ] Uploaded files persist after service restarts.
- [ ] Database backup and recovery procedures are tested.
- [ ] Reverse-proxy upload limits are sufficient.
- [ ] No secrets or private applicant data are exposed.

Use synthetic resumes and synthetic applicant records for public demonstrations.

## 11. Known limitations

The following areas require additional verification before production deployment:

- Actual embedding-model performance and memory consumption on the selected hosting platform.
- Hosting-provider compatibility with the locked runtime dependencies.
- Persistent file storage and database backup configuration.
- Reverse-proxy behavior for large upload requests.
- Production migration procedures for existing databases.
- Real-world evaluation of ranking quality and hiring-related fairness.

Model scores are ranking signals, not probabilities of selection or automated hiring decisions.

## 12. Deployment status

This repository contains deployment preparation scripts, runtime dependency locks, example service configuration, database migrations, and release packaging logic.

Local validation does not establish that a remote deployment is operational.

No public deployment or hosting-provider configuration should be considered complete until the deployment checklist has been verified and the project owner has explicitly approved deployment.