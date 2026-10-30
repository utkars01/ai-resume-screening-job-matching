# AI Resume Screening & Job Matching

An AI-assisted, recruiter-led resume screening and job matching application built with **FastAPI, PostgreSQL, React, and XGBoost**.

The system allows recruiters to manage job openings, upload applicant resumes, review extracted candidate information, rank applicants against job requirements, and maintain job-specific screening history and shortlists.

> **Important:** This project supports recruiter decision-making. Model scores are ranking signals, not hiring decisions or probabilities of selection.

## Features

### Recruiter application

- Recruiter registration and login.
- Password hashing using PBKDF2-HMAC-SHA256.
- Signed HTTP-only session cookies.
- Recruiter-owned job management.
- Job-specific applicant management and screening.
- Applicant batch uploads with per-file processing results.
- Applicant ranking and saved screening history.
- Job-specific shortlist management.
- Ownership checks on protected job and applicant operations.

### Resume processing

- PDF and DOCX resume parsing.
- Maximum file size: 5 MB per resume.
- Maximum batch size: 50 resumes.
- Extraction of skills, education, experience, seniority, and role-related information.
- Missing or uncertain metadata is kept separate rather than automatically treated as known.
- Applicant list responses expose summary information rather than full resume text or filesystem paths.

Scanned PDFs requiring OCR are not currently supported.

### AI-based ranking

- XGBoost-based applicant ranking.
- Job-specific candidate comparison.
- Required and preferred skill information.
- Skill coverage and compatibility inputs.
- A shared 21-feature contract between feature generation and inference.
- Production model: `models/xgboost_ranker.json`.
- Feature definition: `models/ranking_features.txt`.

The application displays raw model scores. These are not calibrated probabilities, and compatibility inputs should not be interpreted as explanations of each feature's contribution to an individual prediction.

## Technology stack

| Component | Technology |
|---|---|
| Backend | Python, FastAPI |
| Database | PostgreSQL |
| ORM | SQLAlchemy |
| Frontend | React, Vite |
| Machine learning | XGBoost |
| Embeddings | Sentence Transformers |
| Resume parsing | PyMuPDF, python-docx |
| Backend testing | pytest |
| Frontend testing | Vitest, Testing Library |

## Project structure

```text
ai-resume-screening-job-matching/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── models/
│   │   ├── schemas/
│   │   └── services/
│   └── migrations/
│       └── versions/
│
├── frontend/
│   ├── src/
│   │   ├── RecruiterPortal.jsx
│   │   └── RecruiterPortal.test.jsx
│   ├── package.json
│   └── vite.config.js
│
├── data/
│   ├── datasets/
│   └── README.md
│
├── models/
│   ├── ranking_features.txt
│   └── xgboost_ranker.json
│
├── scripts/
│   ├── build_release_candidate.py
│   └── migrate_database.py
│
├── tests/
├── deployment/
├── requirements.txt
├── requirements-runtime.in
├── requirements-runtime.lock
├── pytest.ini
├── README.md
└── DEPLOYMENT.md
```

Raw datasets, processed datasets, uploaded resumes, environment files, and local development artifacts are intentionally excluded from version control.

## Prerequisites

Install the following before running the application:

- Python 3.12
- Node.js and npm
- PostgreSQL
- Git

The production model and feature definition are included in the repository. The runtime skill vocabulary and the Sentence Transformers embedding model must also be available locally.

## Installation and local setup

### 1. Clone the repository

```bash
git clone https://github.com/utkars01/ai-resume-screening-job-matching.git

cd ai-resume-screening-job-matching
```

### 2. Set up the Python environment

Windows PowerShell:

```powershell
python -m venv .venv

.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
```

### 3. Configure environment variables

Create a local environment file using the example:

```powershell
Copy-Item backend\.env.example backend\.env
```

Configure the required values, including:

- `DATABASE_URL`: PostgreSQL connection URL.
- `AUTH_SECRET`: a securely generated random secret of at least 32 UTF-8 bytes.
- `DEBUG`: development or release configuration as appropriate.

Never commit `.env` files, database passwords, authentication secrets, or API tokens.

### 4. Prepare PostgreSQL

Create a new, empty PostgreSQL database and configure its connection URL.

Run the versioned migrations:

```powershell
.venv\Scripts\python.exe scripts\migrate_database.py
```

**Database warning:** The migration runner is designed for a new, empty database. It refuses to adopt an existing populated database without migration history. Do not run it against an existing database without a separately reviewed migration and backup plan.

### 5. Prepare the runtime skill vocabulary

The runtime vocabulary is generated from the job-posting dataset. The raw and processed datasets are not included in this repository.

Follow the data preparation instructions in [`data/README.md`](data/README.md). The final runtime file required by the release builder is:

```text
data/processed/skill_master.csv
```

### 6. Install frontend dependencies

```powershell
cd frontend

npm ci

cd ..
```

### 7. Start the backend

From the repository root:

```powershell
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload
```

The API is available at:

```text
http://127.0.0.1:8000
```

During development, API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

### 8. Start the frontend

Open another terminal:

```powershell
cd frontend

npm run dev
```

Open the local Vite URL shown in the terminal.

The frontend development proxy forwards `/api` requests to the local backend. The backend does not currently enable CORS middleware; production deployment should use same-origin frontend and API routing unless an explicit origin allowlist is configured.

## Application workflow

1. Register or sign in as a recruiter.
2. Create a job and provide its requirements.
3. Select the job in the applicant screening interface.
4. Upload PDF or DOCX resumes.
5. Review file-processing outcomes and extracted candidate information.
6. Rank applicants for the selected job.
7. Review ranking results and available qualification information.
8. Save screening history and manage job-specific shortlists.

Recruiters remain responsible for reviewing candidates and making hiring decisions.

## Data and model pipeline

The project includes scripts for:

- Downloading job postings and synthetic resume data.
- Preprocessing job postings and resumes.
- Skill normalization and canonical skill consolidation.
- Building job and resume skill profiles.
- Generating ranking features and relevance labels.
- Preparing job-disjoint training and validation splits.
- Training and evaluating ranking models.
- Generating candidate recommendations and skill-gap information.

### Dataset sources

The project has used these Hugging Face datasets:

- [NextGig-Rocks/global-job-postings-multi-ats](https://huggingface.co/datasets/NextGig-Rocks/global-job-postings-multi-ats)
- [michaelozon/candidate-matching-synthetic](https://huggingface.co/datasets/michaelozon/candidate-matching-synthetic)

Consult the dataset cards and licence terms before downloading, modifying, or redistributing the data. Dataset revisions used for the historical local downloads were not recorded, so those exact historical snapshots cannot be guaranteed.

Raw and processed datasets are not committed to this repository. They must be obtained and prepared locally. See [`data/README.md`](data/README.md) for the data policy and preparation sequence.

### Ranking model

The production inference model is:

```text
models/xgboost_ranker.json
```

Its ordered feature contract is:

```text
models/ranking_features.txt
```

The inference service validates the expected feature structure before generating ranking scores.

Experimental models and temporary training artifacts are not part of the production inference path.

### Evaluation limitations

Automatically generated relevance labels are heuristic labels derived from structured compatibility rules. Agreement with those labels does not establish real-world ranking quality, recruiter agreement, or hiring outcomes.

A meaningful assessment of recruiter usefulness requires appropriately reviewed human judgments and a clearly defined evaluation methodology. Do not treat heuristic metrics as evidence of hiring performance.

## Tests

### Backend

From the repository root:

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Backend tests cover areas including:

- Resume parsing and metadata extraction.
- Authentication and protected endpoints.
- Applicant and job ownership checks.
- Ranking and feature validation.
- Screening history and shortlist operations.

### Frontend

From the `frontend` directory:

```bash
npm test
npm run lint
npm run build
```

The frontend test suite uses Vitest and Testing Library.

## Security and privacy

- Keep secrets in local environment files or a secure deployment secret manager.
- Never commit real resumes, applicant records, or personal information.
- Use HTTPS and secure cookies in production.
- Use durable, access-controlled storage for uploaded resumes.
- Restrict database access and maintain backups.
- Review migration requirements before applying schema changes to an existing database.
- Do not expose database credentials or authentication secrets through frontend environment variables.

Resume documents and applicant information may contain sensitive personal data. Use synthetic data for demonstrations and testing.

## Deployment

Deployment packaging, runtime configuration, migration safeguards, HTTPS routing, model caching, and persistent upload storage are documented in [`DEPLOYMENT.md`](DEPLOYMENT.md).

The release builder packages an allowlisted set of runtime artifacts and generates a release manifest. It requires a verified runtime skill vocabulary and a successful frontend production build.

A release package is not, by itself, proof that a deployment environment has been verified. Review the target hosting platform, persistent storage, database migration baseline, backups, and runtime resource requirements before deployment.

## Current limitations and future improvements

- OCR support for scanned resumes.
- Human-reviewed evaluation of candidate ranking.
- Further validation of embedding inference performance in the target deployment environment.
- Production-grade shared rate limiting for multi-worker or multi-replica deployments.
- Additional deployment and backup/restore verification.

## Project status

The repository contains an implemented recruiter-facing application, backend APIs, database migrations, resume-processing functionality, and a production XGBoost ranking path.

Further evaluation and deployment validation remain necessary before treating the system as a production hiring platform.