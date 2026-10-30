"""Password hashing and signed access tokens using Python's standard library."""

import base64
import hashlib
import hmac
import json
import secrets
import time


PASSWORD_SCHEME = "pbkdf2_sha256"
PASSWORD_ITERATIONS = 600_000
TOKEN_ISSUER = "ai-resume-screening"


def hash_password(password: str) -> str:
    if not isinstance(password, str) or len(password) < 12:
        raise ValueError("Password must contain at least 12 characters.")
    if len(password) > 1024:
        raise ValueError("Password is too long.")

    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_ITERATIONS,
    )
    return "$".join(
        (
            PASSWORD_SCHEME,
            str(PASSWORD_ITERATIONS),
            _b64encode(salt),
            _b64encode(digest),
        )
    )


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        scheme, rounds, salt_text, digest_text = stored_hash.split("$")
        iterations = int(rounds)
        if scheme != PASSWORD_SCHEME or not 100_000 <= iterations <= 1_000_000:
            return False
        salt = _b64decode(salt_text)
        expected = _b64decode(digest_text)
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, iterations
        )
        return hmac.compare_digest(actual, expected)
    except (AttributeError, ValueError, TypeError):
        return False


def create_access_token(
    user_id: int,
    secret: str,
    *,
    expires_in_seconds: int = 43_200,
    now: int | None = None,
) -> str:
    _validate_secret(secret)
    if user_id < 1 or expires_in_seconds < 1:
        raise ValueError("Token subject and lifetime must be positive.")

    issued_at = int(time.time() if now is None else now)
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "iss": TOKEN_ISSUER,
        "sub": str(user_id),
        "iat": issued_at,
        "exp": issued_at + expires_in_seconds,
    }
    signing_input = f"{_json_segment(header)}.{_json_segment(payload)}"
    signature = hmac.new(
        secret.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256
    ).digest()
    return f"{signing_input}.{_b64encode(signature)}"


def read_access_token(token: str, secret: str, *, now: int | None = None) -> int:
    _validate_secret(secret)
    try:
        header_segment, payload_segment, signature_segment = token.split(".")
        signing_input = f"{header_segment}.{payload_segment}"
        expected = hmac.new(
            secret.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256
        ).digest()
        if not hmac.compare_digest(expected, _b64decode(signature_segment)):
            raise ValueError("Invalid token signature.")

        header = json.loads(_b64decode(header_segment))
        payload = json.loads(_b64decode(payload_segment))
        if header.get("alg") != "HS256" or payload.get("iss") != TOKEN_ISSUER:
            raise ValueError("Unsupported token.")

        subject = payload.get("sub", "")
        expires_at = payload.get("exp")
        issued_at = payload.get("iat")
        current_time = int(time.time() if now is None else now)
        if (
            not isinstance(subject, str)
            or not subject.isdecimal()
            or int(subject) < 1
            or not isinstance(expires_at, int)
            or not isinstance(issued_at, int)
            or expires_at <= current_time
            or issued_at > current_time + 60
        ):
            raise ValueError("Expired or invalid token claims.")
        return int(subject)
    except (AttributeError, KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid access token.") from exc


def _validate_secret(secret: str) -> None:
    if not isinstance(secret, str) or len(secret.encode("utf-8")) < 32:
        raise ValueError("AUTH_SECRET must contain at least 32 bytes.")


def _json_segment(value: dict) -> str:
    return _b64encode(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
