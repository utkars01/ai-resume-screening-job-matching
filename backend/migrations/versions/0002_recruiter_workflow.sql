ALTER TABLE users ADD COLUMN password_hash TEXT;
ALTER TABLE jobs ADD CONSTRAINT uq_jobs_id_recruiter UNIQUE (id, recruiter_id);
ALTER TABLE applicants ADD CONSTRAINT uq_applicants_job_id_id UNIQUE (job_id, id);

CREATE TABLE screening_runs (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    job_id INTEGER NOT NULL,
    recruiter_id INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_screening_runs_id_job UNIQUE (id, job_id),
    CONSTRAINT fk_screening_runs_job_owner
        FOREIGN KEY (job_id, recruiter_id)
        REFERENCES jobs (id, recruiter_id) ON DELETE CASCADE
);

CREATE TABLE screening_results (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    screening_run_id BIGINT NOT NULL,
    job_id INTEGER NOT NULL,
    applicant_id INTEGER NOT NULL,
    rank INTEGER NOT NULL CHECK (rank > 0),
    ranking_score DOUBLE PRECISION NOT NULL,
    evidence JSONB NOT NULL DEFAULT '{}'::jsonb,
    CONSTRAINT uq_screening_results_run_applicant
        UNIQUE (screening_run_id, applicant_id),
    CONSTRAINT uq_screening_results_run_rank UNIQUE (screening_run_id, rank),
    CONSTRAINT fk_screening_results_run_job
        FOREIGN KEY (screening_run_id, job_id)
        REFERENCES screening_runs (id, job_id) ON DELETE CASCADE,
    CONSTRAINT fk_screening_results_job_applicant
        FOREIGN KEY (job_id, applicant_id)
        REFERENCES applicants (job_id, id) ON DELETE CASCADE
);

CREATE TABLE applicant_shortlists (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    recruiter_id INTEGER NOT NULL,
    job_id INTEGER NOT NULL,
    applicant_id INTEGER NOT NULL,
    note TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_applicant_shortlists_job_applicant UNIQUE (job_id, applicant_id),
    CONSTRAINT fk_applicant_shortlists_job_owner
        FOREIGN KEY (job_id, recruiter_id)
        REFERENCES jobs (id, recruiter_id) ON DELETE CASCADE,
    CONSTRAINT fk_applicant_shortlists_job_applicant
        FOREIGN KEY (job_id, applicant_id)
        REFERENCES applicants (job_id, id) ON DELETE CASCADE
);
