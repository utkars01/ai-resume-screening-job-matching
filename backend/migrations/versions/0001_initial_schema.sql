-- Frozen initial schema for an empty PostgreSQL database. Generated from

-- the active SQLAlchemy models on 2026-10-28; do not replay on an existing DB.

CREATE TABLE users (
	id SERIAL NOT NULL,
	email VARCHAR(255) NOT NULL,
	full_name VARCHAR(255) NOT NULL,
	role VARCHAR(50) NOT NULL,
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
	PRIMARY KEY (id)
);

CREATE UNIQUE INDEX ix_users_email ON users (email);

CREATE INDEX ix_users_id ON users (id);

CREATE TABLE candidates (
	id SERIAL NOT NULL,
	user_id INTEGER NOT NULL,
	phone VARCHAR(30),
	location VARCHAR(255),
	headline VARCHAR(255),
	summary TEXT,
	role VARCHAR(255),
	seniority VARCHAR(100),
	years_experience INTEGER,
	education VARCHAR(255),
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
	PRIMARY KEY (id),
	UNIQUE (user_id),
	FOREIGN KEY(user_id) REFERENCES users (id)
);

CREATE INDEX ix_candidates_id ON candidates (id);

CREATE TABLE jobs (
	id SERIAL NOT NULL,
	recruiter_id INTEGER NOT NULL,
	title VARCHAR(255) NOT NULL,
	company_name VARCHAR(255) NOT NULL,
	location VARCHAR(255),
	industry VARCHAR(255),
	description TEXT NOT NULL,
	required_skills TEXT,
	preferred_skills TEXT,
	experience_required INTEGER,
	seniority_required VARCHAR(50),
	education_required VARCHAR(255),
	status VARCHAR(50) NOT NULL,
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
	PRIMARY KEY (id),
	FOREIGN KEY(recruiter_id) REFERENCES users (id)
);

CREATE INDEX ix_jobs_id ON jobs (id);

CREATE INDEX ix_jobs_recruiter_id ON jobs (recruiter_id);

CREATE TABLE applicants (
	id SERIAL NOT NULL,
	job_id INTEGER NOT NULL,
	original_filename VARCHAR(255) NOT NULL,
	file_path VARCHAR(500) NOT NULL,
	file_type VARCHAR(20) NOT NULL,
	extracted_text TEXT,
	processing_status VARCHAR(50) NOT NULL,
	role VARCHAR(255),
	seniority VARCHAR(100),
	years_experience INTEGER,
	education VARCHAR(255),
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
	PRIMARY KEY (id),
	FOREIGN KEY(job_id) REFERENCES jobs (id)
);

CREATE INDEX ix_applicants_id ON applicants (id);

CREATE INDEX ix_applicants_job_id ON applicants (job_id);

CREATE TABLE job_ai_profiles (
	id SERIAL NOT NULL,
	job_id INTEGER NOT NULL,
	required_skills JSON NOT NULL,
	preferred_skills JSON NOT NULL,
	embedding_dimension INTEGER,
	embedding_reference TEXT,
	processing_version TEXT NOT NULL,
	processed_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
	PRIMARY KEY (id),
	FOREIGN KEY(job_id) REFERENCES jobs (id)
);

CREATE INDEX ix_job_ai_profiles_id ON job_ai_profiles (id);

CREATE UNIQUE INDEX ix_job_ai_profiles_job_id ON job_ai_profiles (job_id);

CREATE TABLE resumes (
	id SERIAL NOT NULL,
	candidate_id INTEGER NOT NULL,
	original_filename VARCHAR(255) NOT NULL,
	file_path VARCHAR(500) NOT NULL,
	file_type VARCHAR(20) NOT NULL,
	extracted_text TEXT,
	processing_status VARCHAR(50) NOT NULL,
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
	PRIMARY KEY (id),
	FOREIGN KEY(candidate_id) REFERENCES candidates (id)
);

CREATE INDEX ix_resumes_candidate_id ON resumes (candidate_id);

CREATE INDEX ix_resumes_id ON resumes (id);

CREATE TABLE applicant_ai_profiles (
	id SERIAL NOT NULL,
	applicant_id INTEGER NOT NULL,
	extracted_skills JSON NOT NULL,
	embedding_dimension INTEGER,
	embedding_reference TEXT,
	processing_version TEXT NOT NULL,
	processed_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
	PRIMARY KEY (id),
	FOREIGN KEY(applicant_id) REFERENCES applicants (id)
);

CREATE UNIQUE INDEX ix_applicant_ai_profiles_applicant_id ON applicant_ai_profiles (applicant_id);

CREATE INDEX ix_applicant_ai_profiles_id ON applicant_ai_profiles (id);

CREATE TABLE resume_ai_profiles (
	id SERIAL NOT NULL,
	resume_id INTEGER NOT NULL,
	extracted_skills JSON NOT NULL,
	embedding_dimension INTEGER,
	embedding_reference TEXT,
	processing_version TEXT NOT NULL,
	processed_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
	PRIMARY KEY (id),
	FOREIGN KEY(resume_id) REFERENCES resumes (id)
);

CREATE INDEX ix_resume_ai_profiles_id ON resume_ai_profiles (id);

CREATE UNIQUE INDEX ix_resume_ai_profiles_resume_id ON resume_ai_profiles (resume_id);
