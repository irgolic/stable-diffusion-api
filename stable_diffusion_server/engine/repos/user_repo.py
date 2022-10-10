import datetime
from calendar import timegm
from typing import Optional

import jose
from jose import jwt
from passlib.context import CryptContext

from stable_diffusion_server.engine.utils import get_redis
from stable_diffusion_server.models.user import Username, UserInDB, DefaultUsername, UserBase, AuthenticationError, \
    AuthToken, User


class UserRepo:
    def __init__(
        self,
        secret_key: str,
        algorithm: str,
        access_token_expires: datetime.timedelta,
        allow_public_token: bool,
    ):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expires = access_token_expires
        self.allow_public_token = allow_public_token

    def _get_user_by_username(self, username: Username) -> Optional[UserInDB]:
        raise NotImplementedError

    def _put_user(self, user: UserInDB) -> None:
        raise NotImplementedError

    # def get_user_by_username_and_password(self, username: Username, password: str) -> Optional[User]:
    #     user_in_db = self._get_user_by_username(username)
    #     if user_in_db is None:
    #         return None
    #     if not self.pwd_context.verify(password, user_in_db.hashed_password):
    #         raise AuthenticationError("Invalid password")
    #     return User(**user_in_db.dict())

    def create_user(self, user: UserBase, password: str) -> None:
        if not self.allow_public_token and user.username == DefaultUsername:
            raise AuthenticationError("Invalid username")

        if self._get_user_by_username(user.username) is not None:
            raise AuthenticationError(f"User {user.username} already exists")
        self._put_user(UserInDB(
            hashed_password=self.pwd_context.hash(password),
            **user.dict(),
        ))

    def create_public_token(self) -> AuthToken:
        if not self.allow_public_token:
            raise RuntimeError("Public token is not allowed")

        user_in_db = self._get_user_by_username(DefaultUsername)
        if user_in_db is None:
            # default user's password is insecure; the other functions explicitly disallow login with it when disabled
            self.create_user(UserBase(username=DefaultUsername), password="password")
        try:
            return self.create_token_by_username_and_password(DefaultUsername, "password")
        except AuthenticationError:
            raise RuntimeError(f"Failed to create default token as User 'all'")

    def create_token_by_username_and_password(self, username: Username, password: str) -> AuthToken:
        if not self.allow_public_token and username == DefaultUsername:
            raise AuthenticationError("Invalid username")

        user_in_db = self._get_user_by_username(username)
        if user_in_db is None:
            raise AuthenticationError(f"User {username} does not exist")
        if not self.pwd_context.verify(password, user_in_db.hashed_password):
            raise AuthenticationError("Invalid password")

        username = user_in_db.username
        expire = timegm((datetime.datetime.utcnow() + self.access_token_expires).utctimetuple())
        data = {
            "sub": username,
            "exp": expire,
        }

        token_string = jwt.encode(data, self.secret_key, algorithm=self.algorithm)
        return AuthToken(access_token=token_string, token_type="bearer")

    def must_get_user_by_token(self, token: str) -> User:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
        except jose.JWTError:
            raise AuthenticationError("Invalid token")

        expire = payload.get("exp")
        if expire is None:
            raise AuthenticationError("Invalid token")

        if datetime.datetime.fromtimestamp(expire) < datetime.datetime.utcnow():
            raise AuthenticationError("Token expired")

        username = payload.get("sub")
        if username is None or (not self.allow_public_token and username == DefaultUsername):
            raise AuthenticationError("Invalid token")

        # very unlikely for token not to be individual per retrieval from the same user,
        # because of the 'expire' part. so just encode it deterministically to make a session id
        # TODO revisit this
        session_id = jwt.encode(
            {'hehe': token + 'ARBITREAARY_PHR4SE+WOOHOO'},
            self.secret_key,
            algorithm=self.algorithm
        )

        user_in_db = self._get_user_by_username(username)
        if user_in_db is None:
            raise AuthenticationError("Invalid token")
        return User(
            **user_in_db.dict(),
            session_id=session_id,
        )


class InMemoryUserRepo(UserRepo):
    _in_memory_users = {}

    def _get_user_by_username(self, username: Username) -> Optional[UserInDB]:
        return self._in_memory_users.get(username)

    def _put_user(self, user: UserInDB) -> None:
        self._in_memory_users[user.username] = user


class RedisUserRepo(UserRepo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.redis = get_redis()

    def _get_user_by_username(self, username: Username) -> Optional[UserInDB]:
        json_user = self.redis.hget('user', username)
        if json_user is None:
            return None
        return UserInDB.parse_raw(json_user)

    def _put_user(self, user: UserInDB) -> None:
        self.redis.hset('user', user.username, user.json())
