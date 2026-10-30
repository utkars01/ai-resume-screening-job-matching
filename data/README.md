# Project Data Documentation

This directory contains the data acquisition, preprocessing, skill normalization, feature engineering, and evaluation resources for the **AI Resume Screening & Job Matching System**.

## 1. Directory Structure

```text
data/
├── raw/                 # Original downloaded datasets
├── processed/           # Cleaned and transformed datasets
└── datasets/            # Data processing and ML pipeline scripts
```

Generated datasets and intermediate files are maintained locally and are not intended for public GitHub commits.

## 2. Dataset Sources and Licenses

### Resume Dataset

- **Dataset:** Candidate Matching Synthetic
- **Source:** Hugging Face – `michaelozon/candidate-matching-synthetic`
- **License:** MIT
- **Purpose:** Resume profiles, skills, experience, education, and candidate information for matching experiments.
- **Raw file:** `raw/resumes_raw.csv`
- **Processed file:** `processed/resumes_processed.csv`

### Job Postings Dataset

- **Dataset:** Global Job Postings Multi-ATS
- **Source:** Hugging Face – `NextGig-Rocks/global-job-postings-multi-ats`
- **License:** CC BY 4.0
- **Attribution:** NextGig
- **Snapshot used:** June 2026
- **Purpose:** Job descriptions, required skills, job titles, industries, experience requirements, and related job information.
- **Raw file:** `raw/job_postings_raw.csv`
- **Processed file:** `processed/job_postings_processed.csv`

The original dataset licenses and attribution requirements remain applicable to their respective source data. Consult the original dataset cards and license terms before redistribution.

The exact historical dataset revisions used were not recorded. Do not assume that downloading the current versions will reproduce the original project results exactly.

## 3. Data Processing Policy

The following principles apply:

1. Raw datasets must never be modified directly.
2. Preprocessing outputs must be written to `processed/`.
3. Intermediate datasets must remain separate from original source files.
4. Skill normalization and consolidation must use consistent canonical mappings.
5. Training, validation, and test splits must avoid unintended data leakage.
6. Generated files must not overwrite validated production model artifacts.
7. Personally identifiable or confidential applicant information must not be committed to public repositories.

## 4. Processing Pipeline

The job data processing workflow includes:

1. Download the original job postings dataset.
2. Preprocess job descriptions and structured fields.
3. Extract and normalize job skills.
4. Consolidate skill variants into canonical skill names.
5. Calculate skill frequency statistics.
6. Build the consolidated skill vocabulary.
7. Generate job and resume skill profiles.
8. Construct ranking features and relevance labels.
9. Prepare ranking splits.
10. Train and evaluate the ranking model.

Relevant scripts are maintained under `data/datasets/`.

## 5. Generated Data

The processing workflow may generate intermediate and final artifacts, including:

- Processed resume and job posting datasets
- Normalized and canonical job skills
- Skill frequency statistics
- Skill master vocabulary
- Resume and job skill profiles
- Ranking features and relevance labels
- Training and evaluation datasets
- Embedding-related artifacts

These generated files are local pipeline outputs. They should not be committed to GitHub unless a specific artifact has been reviewed and explicitly approved for publication.

## 6. Data Privacy and Repository Policy

The public repository should contain:

- Dataset source references and attribution
- Data processing scripts
- Documentation
- Model and evaluation code that is approved for publication

The public repository should not contain:

- Raw or processed CSV datasets
- Private applicant or recruiter records
- Uploaded resumes or documents
- Database exports or backups
- Credentials, environment files, or authentication secrets
- Unreviewed intermediate datasets

Synthetic data should be used for demonstrations and testing where appropriate. Production applicant data must remain protected.

## 7. Reproducibility

Reproducing the complete pipeline requires access to the original datasets, compatible dependencies, and the expected processing inputs.

Because the exact historical dataset revisions were not recorded, results may differ when the datasets are downloaded again.

Refer to the root `README.md` for project setup and the `DEPLOYMENT.md` file for deployment requirements.

## 8. License and Attribution

Dataset users are responsible for following the respective source licenses:

- Candidate Matching Synthetic – MIT
- Global Job Postings Multi-ATS – CC BY 4.0, attribution to NextGig

The project code and dataset licenses are separate. The dataset licenses do not automatically determine the license of this project's source code.