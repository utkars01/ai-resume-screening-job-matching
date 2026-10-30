from enum import Enum

from pydantic import BaseModel, ConfigDict, EmailStr


class UserRole(str, Enum):
    CANDIDATE = "candidate"
    RECRUITER = "recruiter"
    ADMIN = "admin"


class UserCreate(BaseModel):
    email: EmailStr
    full_name: str
    role: UserRole = UserRole.CANDIDATE


class UserResponse(BaseModel):
    id: int
    email: EmailStr
    full_name: str
    role: UserRole

    model_config = ConfigDict(from_attributes=True)