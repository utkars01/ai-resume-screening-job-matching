from collections import defaultdict, deque
from threading import Lock
import time

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.auth import get_current_recruiter
from backend.app.config import settings
from backend.app.database.session import get_db
from backend.app.models.user import User
from backend.app.schemas.auth import AuthenticatedUser, RecruiterLogin, RecruiterRegister
from backend.app.services.auth_security import (
    create_access_token,
    hash_password,
    verify_password,
)


router = APIRouter(prefix="/api/auth", tags=["Recruiter Authentication"])
COOKIE_NAME = "recruiter_session"
_LOGIN_WINDOW_SECONDS = 600
_LOGIN_LIMIT = 10
_REGISTER_WINDOW_SECONDS = 3600
_REGISTER_LIMIT = 5
_attempts = defaultdict(deque)
_attempts_lock = Lock()


def _rate_limit(request: Request, *, registering: bool) -> None:
    client_host = request.client.host if request.client else "unknown"
    key = ("register" if registering else "login", client_host)
    window = _REGISTER_WINDOW_SECONDS if registering else _LOGIN_WINDOW_SECONDS
    limit = _REGISTER_LIMIT if registering else _LOGIN_LIMIT
    now = time.monotonic()

    with _attempts_lock:
        events = _attempts[key]
        while events and now - events[0] >= window:
            events.popleft()
        if len(events) >= limit:
            retry_after = max(1, int(window - (now - events[0])))
            raise HTTPException(
                status_code=429,
                detail="Too many account requests. Try again later.",
                headers={"Retry-After": str(retry_after)},
            )
        events.append(now)

        # Keep the in-process guard bounded if the service sees many clients.
        if len(_attempts) > 10_000:
            expired_before = now - max(_LOGIN_WINDOW_SECONDS, _REGISTER_WINDOW_SECONDS)
            for old_key in list(_attempts)[:1000]:
                old_events = _attempts[old_key]
                while old_events and old_events[0] < expired_before:
                    old_events.popleft()
                if not old_events:
                    _attempts.pop(old_key, None)


def _configured_secret() -> str:
    if len(settings.auth_secret.encode("utf-8")) < 32:
        raise HTTPException(
            status_code=503,
            detail="Recruiter authentication is not configured on this server.",
        )
    return settings.auth_secret


def _require_password_schema(db: Session) -> None:
    exists = db.execute(
        text(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_schema = current_schema() "
            "AND table_name = 'users' AND column_name = 'password_hash'"
        )
    ).scalar()
    if not exists:
        raise HTTPException(
            status_code=503,
            detail="Recruiter accounts require the reviewed database schema update.",
        )


def _issue_session(response: Response, user_id: int, secret: str) -> None:
    lifetime = settings.auth_token_ttl_minutes * 60
    token = create_access_token(
        user_id,
        secret,
        expires_in_seconds=lifetime,
    )
    response.set_cookie(
        COOKIE_NAME,
        token,
        max_age=lifetime,
        httponly=True,
        secure=settings.auth_cookie_secure,
        samesite="lax",
        path="/",
    )


@router.post("/register", response_model=AuthenticatedUser, status_code=201)
def register_recruiter(
    data: RecruiterRegister,
    response: Response,
    request: Request,
    db: Session = Depends(get_db),
):
    _rate_limit(request, registering=True)
    secret = _configured_secret()
    _require_password_schema(db)
    email = str(data.email).strip().lower()
    existing_account = db.execute(
        text("SELECT 1 FROM users WHERE lower(email) = :email"),
        {"email": email},
    ).scalar()
    if existing_account:
        raise HTTPException(
            status_code=409,
            detail="An account with this email already exists.",
        )

    password_hash = hash_password(data.password)

    try:
        row = db.execute(
            text(
                "INSERT INTO users "
                "(email, full_name, role, password_hash, created_at) "
                "VALUES (:email, :full_name, 'recruiter', :password_hash, CURRENT_TIMESTAMP) "
                "RETURNING id, email, full_name, role"
            ),
            {
                "email": email,
                "full_name": data.full_name.strip(),
                "password_hash": password_hash,
            },
        ).mappings().one()
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail="An account with this email already exists.",
        )

    _issue_session(response, row["id"], secret)
    return row


@router.post("/login", response_model=AuthenticatedUser)
def login_recruiter(
    data: RecruiterLogin,
    response: Response,
    request: Request,
    db: Session = Depends(get_db),
):
    _rate_limit(request, registering=False)
    secret = _configured_secret()
    _require_password_schema(db)
    row = db.execute(
        text(
            "SELECT id, email, full_name, role, password_hash FROM users "
            "WHERE lower(email) = :email AND role = 'recruiter'"
        ),
        {"email": str(data.email).strip().lower()},
    ).mappings().first()

    if row is None or not row["password_hash"] or not verify_password(
        data.password, row["password_hash"]
    ):
        raise HTTPException(status_code=401, detail="Email or password is incorrect.")

    _issue_session(response, row["id"], secret)
    return {
        "id": row["id"],
        "email": row["email"],
        "full_name": row["full_name"],
        "role": row["role"],
    }


@router.post("/logout", status_code=204)
def logout_recruiter(response: Response):
    response.delete_cookie(
        COOKIE_NAME,
        httponly=True,
        secure=settings.auth_cookie_secure,
        samesite="lax",
        path="/",
    )


@router.get("/me", response_model=AuthenticatedUser)
def current_recruiter(
    recruiter: User = Depends(get_current_recruiter),
):
    return recruiter
