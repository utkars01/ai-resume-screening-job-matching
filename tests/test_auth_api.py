from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from backend.app.api import auth as auth_api
from backend.app.auth import get_current_recruiter
from backend.app.config import settings
from backend.app.database.session import get_db
from backend.app.main import app
from backend.app.models.user import User
from backend.app.services.auth_security import create_access_token, hash_password


SECRET = "test-only-auth-secret-that-is-long-enough-32-bytes"


def test_registration_fails_closed_without_password_column(monkeypatch):
    monkeypatch.setattr(settings, "auth_secret", SECRET)
    db = MagicMock()
    db.execute.return_value.scalar.return_value = None

    def override_db():
        yield db

    app.dependency_overrides[get_db] = override_db
    try:
        response = TestClient(app).post(
            "/api/auth/register",
            json={
                "email": "recruiter@example.com",
                "full_name": "Local Recruiter",
                "password": "strong-test-password-123",
            },
        )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 503
    assert "schema update" in response.json()["detail"]
    db.commit.assert_not_called()


def test_registration_rejects_case_insensitive_existing_email(monkeypatch):
    monkeypatch.setattr(settings, "auth_secret", SECRET)
    schema_check = MagicMock()
    schema_check.scalar.return_value = 1
    existing_account = MagicMock()
    existing_account.scalar.return_value = 1
    db = MagicMock()
    db.execute.side_effect = [schema_check, existing_account]

    def override_db():
        yield db

    app.dependency_overrides[get_db] = override_db
    monkeypatch.setattr(
        auth_api,
        "hash_password",
        lambda _: (_ for _ in ()).throw(AssertionError("password hashing should not run")),
    )
    try:
        response = TestClient(app).post(
            "/api/auth/register",
            json={
                "email": "RECRUITER@example.com",
                "full_name": "Local Recruiter",
                "password": "strong-test-password-123",
            },
        )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 409
    assert response.json()["detail"] == "An account with this email already exists."
    db.commit.assert_not_called()


def test_registration_creates_recruiter_with_hashed_password_and_cookie(monkeypatch):
    monkeypatch.setattr(settings, "auth_secret", SECRET)
    schema_check = MagicMock()
    schema_check.scalar.return_value = 1
    duplicate_check = MagicMock()
    duplicate_check.scalar.return_value = None
    inserted = MagicMock()
    inserted.mappings.return_value.one.return_value = {
        "id": 24,
        "email": "recruiter@example.com",
        "full_name": "Local Recruiter",
        "role": "recruiter",
    }
    db = MagicMock()
    db.execute.side_effect = [schema_check, duplicate_check, inserted]

    def override_db():
        yield db

    app.dependency_overrides[get_db] = override_db
    password = "strong-test-password-123"
    try:
        response = TestClient(app).post(
            "/api/auth/register",
            json={
                "email": "RECRUITER@example.com",
                "full_name": "Local Recruiter",
                "password": password,
            },
        )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 201
    assert response.json()["role"] == "recruiter"
    cookie = response.headers["set-cookie"]
    assert "HttpOnly" in cookie
    assert "SameSite=lax" in cookie
    insert_parameters = db.execute.call_args_list[-1].args[1]
    insert_statement = str(db.execute.call_args_list[-1].args[0])
    assert "created_at" in insert_statement
    assert "CURRENT_TIMESTAMP" in insert_statement
    stored_hash = insert_parameters["password_hash"]
    assert insert_parameters["email"] == "recruiter@example.com"
    assert insert_parameters["full_name"] == "Local Recruiter"
    assert password not in stored_hash
    assert stored_hash.startswith("pbkdf2_sha256$")
    assert db.commit.called


def test_login_verifies_password_hash_and_issues_cookie(monkeypatch):
    monkeypatch.setattr(settings, "auth_secret", SECRET)
    schema_check = MagicMock()
    schema_check.scalar.return_value = 1
    account = MagicMock()
    account.mappings.return_value.first.return_value = {
        "id": 25,
        "email": "owner@example.com",
        "full_name": "Job Owner",
        "role": "recruiter",
        "password_hash": hash_password("strong-test-password-123"),
    }
    db = MagicMock()
    db.execute.side_effect = [schema_check, account]

    def override_db():
        yield db

    app.dependency_overrides[get_db] = override_db
    try:
        response = TestClient(app).post(
            "/api/auth/login",
            json={
                "email": "OWNER@example.com",
                "password": "strong-test-password-123",
            },
        )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200
    assert response.json()["id"] == 25
    assert response.json()["role"] == "recruiter"
    assert "HttpOnly" in response.headers["set-cookie"]
    db.commit.assert_not_called()


def test_signed_cookie_authenticates_only_recruiter(monkeypatch):
    monkeypatch.setattr(settings, "auth_secret", SECRET)
    recruiter = SimpleNamespace(
        id=19,
        email="owner@example.com",
        full_name="Job Owner",
        role="recruiter",
    )
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = recruiter

    def override_db():
        yield db

    app.dependency_overrides[get_db] = override_db
    token = create_access_token(19, SECRET)
    try:
        response = TestClient(app).get(
            "/api/auth/me",
        cookies={"recruiter_session": token},
        )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200
    assert response.json() == {
        "id": 19,
        "email": "owner@example.com",
        "full_name": "Job Owner",
        "role": "recruiter",
    }
    assert db.query.call_args.args == (User,)
