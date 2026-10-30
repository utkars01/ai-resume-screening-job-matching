from backend.app.config import Settings


def test_release_environment_mode_disables_debug():
    settings = Settings(
        database_url="postgresql://example",
        debug="release",
        auth_secret="production-secret-with-at-least-32-bytes",
        auth_cookie_secure=True,
    )
    assert settings.debug is False


def test_development_environment_mode_enables_debug():
    settings = Settings(database_url="postgresql://example", debug="development")
    assert settings.debug is True


def test_production_requires_strong_secret_and_secure_cookie():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="AUTH_SECRET"):
        Settings(database_url="postgresql://example", debug=False, auth_secret="")

    with pytest.raises(ValidationError, match="AUTH_COOKIE_SECURE"):
        Settings(
            database_url="postgresql://example",
            debug=False,
            auth_secret="production-secret-with-at-least-32-bytes",
            auth_cookie_secure=False,
        )

    with pytest.raises(ValidationError, match="EMBEDDING_LOCAL_FILES_ONLY"):
        Settings(
            database_url="postgresql://example",
            debug=False,
            auth_secret="production-secret-with-at-least-32-bytes",
            auth_cookie_secure=True,
            embedding_local_files_only=False,
        )


def test_upload_storage_and_embedding_revision_are_configurable(tmp_path):
    settings = Settings(
        database_url="postgresql://example",
        upload_storage_dir=tmp_path,
        embedding_model_revision="a" * 40,
    )
    assert settings.upload_storage_dir == tmp_path
    assert settings.embedding_model_revision == "a" * 40
