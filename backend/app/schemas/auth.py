from pydantic import BaseModel, ConfigDict, EmailStr, Field


class RecruiterRegister(BaseModel):
    email: EmailStr
    full_name: str = Field(min_length=1, max_length=255)
    password: str = Field(min_length=12, max_length=1024)

    model_config = ConfigDict(extra="forbid")


class RecruiterLogin(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=1024)

    model_config = ConfigDict(extra="forbid")


class AuthenticatedUser(BaseModel):
    id: int
    email: EmailStr
    full_name: str
    role: str
