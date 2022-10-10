import os
import datetime
from typing import Type, Union

import pydantic
import yaml
from fastapi.openapi.utils import get_openapi

from fastapi import FastAPI, Depends, websockets, Query, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from starlette import status

from stable_diffusion_server.engine.repos.blob_repo import BlobRepo
from stable_diffusion_server.engine.repos.messaging_repo import MessagingRepo
from stable_diffusion_server.engine.repos.user_repo import UserRepo
from stable_diffusion_server.engine.services.event_service import EventListener
from stable_diffusion_server.engine.services.task_service import TaskService
from stable_diffusion_server.models.image import GeneratedImage, Image
from stable_diffusion_server.models.model import Model
from stable_diffusion_server.models.params import Params
from stable_diffusion_server.models.task import Txt2ImgTask, TaskId, Img2ImgTask
from stable_diffusion_server.models.user import UserBase, AuthenticationError, User, AuthToken


class AppConfig(pydantic.BaseModel):
    blob_repo_class: Type[BlobRepo]
    messaging_repo_class: Type[MessagingRepo]
    user_repo: Type[UserRepo]

    SECRET_KEY = os.environ["SECRET_KEY"]
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 240

    ENABLE_PUBLIC_TOKEN: bool = bool(os.getenv('ENABLE_PUBLIC_TOKEN', False))
    ENABLE_SIGNUP: bool = bool(os.getenv('ENABLE_SIGNUP', False))


def create_app(app_config: AppConfig) -> FastAPI:
    app = FastAPI(
        title="Stable Diffusion Server",
    )

    ###
    # Authentication
    ###

    async def construct_user_repo():
        return app_config.user_repo(
            secret_key=app_config.SECRET_KEY,
            algorithm=app_config.ALGORITHM,
            access_token_expires=datetime.timedelta(minutes=app_config.ACCESS_TOKEN_EXPIRE_MINUTES),
            allow_public_token=app_config.ENABLE_PUBLIC_TOKEN,
        )

    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    async def get_user(token: str = Depends(oauth2_scheme),
                       user_repo: UserRepo = Depends(construct_user_repo)) -> User:
        try:
            return user_repo.must_get_user_by_token(token)
        except AuthenticationError:
            raise credentials_exception

    @app.post("/token", response_model=AuthToken)
    async def login_access_token(form_data: OAuth2PasswordRequestForm = Depends(),
                                 user_repo: UserRepo = Depends(construct_user_repo)) -> AuthToken:
        try:
            return user_repo.create_token_by_username_and_password(form_data.username, form_data.password)
        except AuthenticationError:
            raise credentials_exception

    if app_config.ENABLE_PUBLIC_TOKEN:
        @app.post("/token/all", response_model=AuthToken)
        async def public_access_token(user_repo: UserRepo = Depends(construct_user_repo)) -> AuthToken:
            return user_repo.create_public_token()

    if app_config.ENABLE_SIGNUP:
        @app.post("/user/{username}", response_model=UserBase)
        async def signup(username: str,
                         password: str,
                         user_repo: UserRepo = Depends(construct_user_repo)) -> UserBase:
            user = UserBase(
                username=username,
            )
            user_repo.create_user(user, password)
            return user

    ###
    # Engine
    ###

    async def construct_messaging_repo() -> MessagingRepo:
        return app_config.messaging_repo_class()

    async def construct_task_service(
        messaging_repo: MessagingRepo = Depends(construct_messaging_repo),
    ):
        return TaskService(
            messaging_repo=messaging_repo,
        )

    async def construct_event_listener(
        messaging_repo: MessagingRepo = Depends(construct_messaging_repo),
    ):
        return EventListener(
            messaging_repo=messaging_repo
        )

    ###
    # API
    ###

    @app.get("/models", response_model=list[Model])
    async def models() -> list[Model]:
        return []

    @app.post("/txt2img", response_model=TaskId)
    async def txt2img(
        parameters: Params,
        task_service: TaskService = Depends(construct_task_service),
        user: User = Depends(get_user),
    ) -> TaskId:
        task = Txt2ImgTask(
            task_type="txt2img",
            params=parameters,
            session_id=user.session_id,
        )
        task_service.push_task(task)
        return task.task_id

    @app.post("/img2img", response_model=TaskId)
    async def img2img(
        parameters: Params,
        image: Image,
        task_service: TaskService = Depends(construct_task_service),
        user: User = Depends(get_user),
    ) -> TaskId:
        task = Img2ImgTask(
            task_type="img2img",
            params=parameters,
            image=image,
            session_id=user.session_id,
        )
        task_service.push_task(task)
        return task.task_id

    ###
    # Websocket
    ###

    async def get_token(
        websocket: websockets.WebSocket,
        token: Union[str, None] = Query(default=None),
    ) -> str:
        if token is None:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            raise HTTPException(status_code=status.WS_1008_POLICY_VIOLATION, detail="No session or token")
        return token

    async def get_ws_user(
        websocket: websockets.WebSocket,
        token: str = Depends(get_token),
        user_repo: UserRepo = Depends(construct_user_repo),
    ) -> User:
        try:
            return user_repo.must_get_user_by_token(token)
        except AuthenticationError:
            await websocket.close(code=status.HTTP_401_UNAUTHORIZED)
            raise HTTPException(status_code=401, detail="User not found")

    @app.websocket('/events')
    async def websocket_endpoint(
        websocket: websockets.WebSocket,
        user: User = Depends(get_ws_user),
        event_listener: EventListener = Depends(construct_event_listener),
    ):
        await websocket.accept()
        async for session_id, event in event_listener.listen():
            if session_id != user.session_id:
                continue
            await websocket.send_json(event.json())
        await websocket.close()

    ###
    # OpenAPI
    ###

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        openapi_version=app.openapi_version,
        description=app.description,
        routes=app.routes,
        # openapi_prefix=app.openapi_prefix,
    )
    with open('openapi.yml', 'w') as f:
        yaml.dump(openapi_schema, f)
    app.openapi_schema = openapi_schema

    return app
