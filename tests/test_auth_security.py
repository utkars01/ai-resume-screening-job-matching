import pytest

from backend.app.services.auth_security import (
    create_access_token,
    hash_password,
    read_access_token,
    verify_password,
)


SECRET = "local-test-secret-that-is-long-enough-32-bytes"


def test_password_hash_verifies_without_storing_plaintext():
    password = "A-strong-recruiter-password-2026"
    stored = hash_password(password)

    assert password not in stored
    assert verify_password(password, stored)
    assert not verify_password("a different password", stored)


def test_password_hash_rejects_short_passwords():
    with pytest.raises(ValueError, match="12 characters"):
        hash_password("short")


def test_signed_access_token_round_trip_and_expiry():
    token = create_access_token(17, SECRET, expires_in_seconds=60, now=1000)

    assert read_access_token(token, SECRET, now=1059) == 17
    with pytest.raises(ValueError, match="claims"):
        read_access_token(token, SECRET, now=1060)


def test_signed_access_token_rejects_tampering_and_wrong_secret():
    token = create_access_token(17, SECRET, now=1000)
    tampered = token[:-1] + ("A" if token[-1] != "A" else "B")

    with pytest.raises(ValueError, match="signature"):
        read_access_token(tampered, SECRET, now=1001)
    with pytest.raises(ValueError, match="signature"):
        read_access_token(token, "another-secret-that-is-also-at-least-32-bytes", now=1001)


def test_token_requires_a_strong_configured_secret():
    with pytest.raises(ValueError, match="32 bytes"):
        create_access_token(17, "too-short")
