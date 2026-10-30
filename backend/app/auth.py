from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from backend.app.database.session import get_db
from backend.app.models.user import User
from backend.app.services.auth_security import read_access_token
from backend.app.config import settings


bearer_scheme = HTTPBearer(auto_error=False)


def get_current_recruiter(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    token = (
        credentials.credentials
        if credentials is not None
        else request.cookies.get("recruiter_session")
    )
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Recruiter authentication is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        user_id = read_access_token(token, settings.auth_secret)
    except ValueError:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired recruiter session.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    recruiter = (
        db.query(User)
        .filter(User.id == user_id, User.role == "recruiter")
        .first()
    )
    if recruiter is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired recruiter session.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return recruiter
