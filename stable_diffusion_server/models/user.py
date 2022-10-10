from typing import Optional
from typing_extensions import TypeAlias

import pydantic

Username: TypeAlias = str
SessionId: TypeAlias = str

DefaultUsername: Username = "all"


class AuthenticationError(Exception):
    pass


class AuthToken(pydantic.BaseModel):
    access_token: str
    token_type: str


class UserBase(pydantic.BaseModel):
    username: Username


class User(UserBase):
    session_id: SessionId


class UserInDB(UserBase):
    hashed_password: str
