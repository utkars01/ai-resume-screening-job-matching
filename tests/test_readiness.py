from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from backend.app.database.session import get_db
from backend.app.main import app, embedding_service


def test_readiness_requires_preloaded_embedding_model(monkeypatch):
    monkeypatch.setattr(embedding_service, "_model", None)
    response = TestClient(app).get("/ready")
    assert response.status_code == 503
    assert response.json() == {"detail": "Embedding model is not ready."}


def test_readiness_checks_database_without_exposing_error(monkeypatch):
    monkeypatch.setattr(embedding_service, "_model", SimpleNamespace())
    db = MagicMock()
    db.execute.side_effect = RuntimeError("sensitive database diagnostic")

    def override_db():
        yield db

    app.dependency_overrides[get_db] = override_db
    try:
        response = TestClient(app).get("/ready")
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 503
    assert response.json() == {"detail": "Database is not ready."}
    assert "sensitive database diagnostic" not in response.text


def test_readiness_passes_when_database_and_embedding_are_ready(monkeypatch):
    monkeypatch.setattr(embedding_service, "_model", SimpleNamespace())
    db = MagicMock()

    def override_db():
        yield db

    app.dependency_overrides[get_db] = override_db
    try:
        response = TestClient(app).get("/ready")
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
    db.execute.assert_called_once()


def test_application_lifespan_preloads_embedding_model(monkeypatch):
    preload = MagicMock()
    monkeypatch.setattr(embedding_service, "preload", preload)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
    preload.assert_called_once()
