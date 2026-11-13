from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI Resume Screening & Job Matching API"
    app_version: str = "1.0.0"
    debug: bool = False
    database_url: str
    auth_secret: str = ""
    auth_token_ttl_minutes: int = Field(default=720, ge=5, le=1440)
    auth_cookie_secure: bool = True
    upload_storage_dir: Path = Path("data/uploads")
    embedding_model_revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    embedding_local_files_only: bool = True
    cors_origins: str = ""
    serve_frontend: bool = True

    @model_validator(mode="after")
    def validate_production_security(self):
        if not self.debug:
            if len(self.auth_secret.encode("utf-8")) < 32:
                raise ValueError("AUTH_SECRET must contain at least 32 UTF-8 bytes in production")
            if not self.auth_cookie_secure:
                raise ValueError("AUTH_COOKIE_SECURE must be true in production")
            if not self.embedding_local_files_only:
                raise ValueError("EMBEDDING_LOCAL_FILES_ONLY must be true in production")
        return self

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug_mode(cls, value):
        # Existing local configurations may express this setting as an
        # environment name (for example, "release") instead of a Boolean.
        if isinstance(value, bool):
            return value

        normalized = str(value).strip().lower()
        if normalized in {"release", "production", "prod", "false", "0", "no", "off"}:
            return False
        if normalized in {"development", "dev", "true", "1", "yes", "on"}:
            return True

        raise ValueError("DEBUG must be a Boolean or a development/release mode")

    model_config = SettingsConfigDict(
        env_file="backend/.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
