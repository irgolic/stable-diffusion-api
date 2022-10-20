import asyncio
import os
import datetime
import uuid
from collections import defaultdict
from typing import Type, Union, Optional, AsyncGenerator

import bcrypt
import pydantic
import yaml
from fastapi.openapi.utils import get_openapi

from fastapi import FastAPI, Depends, websockets, Query, HTTPException, File, UploadFile
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from starlette import status
from starlette.responses import Response

from stable_diffusion_server.engine.repos.blob_repo import BlobRepo, BlobId
from stable_diffusion_server.engine.repos.key_value_repo import KeyValueRepo
from stable_diffusion_server.engine.repos.messaging_repo import MessagingRepo
from stable_diffusion_server.engine.repos.user_repo import UserRepo
from stable_diffusion_server.engine.services.event_service import EventListener, EventService
from stable_diffusion_server.engine.services.status_service import StatusService
from stable_diffusion_server.engine.services.task_service import TaskService
from stable_diffusion_server.models.blob import Blob
from stable_diffusion_server.models.events import EventUnion, FinishedEvent, CancelledEvent
from stable_diffusion_server.models.params import Txt2ImgParams, Img2ImgParams, ParamsUnion
from stable_diffusion_server.models.task import TaskId, Task
from stable_diffusion_server.models.user import UserBase, AuthenticationError, User, AuthToken


class AppConfig(pydantic.BaseModel):
    blob_repo_class: Type[BlobRepo]
    key_value_repo: Type[KeyValueRepo]
    messaging_repo_class: Type[MessagingRepo]
    user_repo: Type[UserRepo]

    SECRET_KEY: str = pydantic.Field(default_factory=lambda: os.environ["SECRET_KEY"])
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 240

    PRINT_LINK_WITH_TOKEN: bool = pydantic.Field(default_factory=lambda: os.environ["PRINT_LINK_WITH_TOKEN"] == "1")
    ENABLE_PUBLIC_ACCESS: bool = pydantic.Field(default_factory=lambda: os.environ["ENABLE_PUBLIC_ACCESS"] == "1")
    ENABLE_SIGNUP: bool = pydantic.Field(default_factory=lambda: os.environ["ENABLE_SIGNUP"] == "1")


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
            allow_public_token=app_config.ENABLE_PUBLIC_ACCESS,
        )

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # Optional bearer token handler
    optional_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

    async def get_user(
        header_token: Optional[str] = Depends(optional_oauth2_scheme),
        query_token: Optional[str] = Query(default=None, alias="token"),
        user_repo: UserRepo = Depends(construct_user_repo)
    ) -> User:
        token = header_token or query_token

        if token is None:
            if not app_config.ENABLE_PUBLIC_ACCESS:
                raise credentials_exception

            # create an ephemeral public user session
            # no token means no way to receive events on websocket (auth per session identified by token)
            # can still poll task status and use blobs though (auth by username, i.e. "all")
            token = user_repo.create_public_token().access_token

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

    @app.post("/token/all", response_model=AuthToken)
    async def public_access_token(user_repo: UserRepo = Depends(construct_user_repo)) -> AuthToken:
        if not app_config.ENABLE_PUBLIC_ACCESS:
            raise HTTPException(status_code=403, detail="Public token is disabled")
        return user_repo.create_public_token()

    @app.post("/user/{username}", response_model=UserBase)
    async def signup(username: str,
                     password: str,
                     user_repo: UserRepo = Depends(construct_user_repo)) -> UserBase:
        if not app_config.ENABLE_SIGNUP:
            raise HTTPException(status_code=403, detail="Signup is disabled")
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

    async def construct_key_value_repo() -> KeyValueRepo:
        return app_config.key_value_repo()

    async def construct_status_service(
        key_value_repo: KeyValueRepo = Depends(construct_key_value_repo),
    ) -> StatusService:
        return StatusService(
            key_value_repo=key_value_repo,
        )

    async def construct_event_service(
        messaging_repo: MessagingRepo = Depends(construct_messaging_repo),
        status_service: StatusService = Depends(construct_status_service),
    ) -> EventService:
        return EventService(
            messaging_repo=messaging_repo,
            status_service=status_service,
        )

    async def construct_task_service(
        messaging_repo: MessagingRepo = Depends(construct_messaging_repo),
        event_service: EventService = Depends(construct_event_service),
        status_service: StatusService = Depends(construct_status_service),
    ):
        return TaskService(
            messaging_repo=messaging_repo,
            event_service=event_service,
            status_service=status_service,
        )

    async def construct_blob_repo() -> BlobRepo:
        return app_config.blob_repo_class()

    ###
    # Event listener
    ###

    queues_by_session_id: dict[str, list[asyncio.Queue]] = defaultdict(list)
    queues_by_task_id: dict[TaskId, list[asyncio.Queue]] = defaultdict(list)

    async def event_listener():
        listener = EventListener(
            messaging_repo=app_config.messaging_repo_class(),
        )
        async for session_id, event in listener.listen():
            for queue in queues_by_session_id[session_id]:
                await queue.put(event)
            for queue in queues_by_task_id[event.task_id]:
                await queue.put(event)

    @app.on_event("startup")
    async def startup_event():
        asyncio.create_task(event_listener())

    async def subscribe_to_session(session_id: str) -> AsyncGenerator[EventUnion, None]:
        queue = asyncio.Queue()
        queues_by_session_id[session_id].append(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            queues_by_session_id[session_id].remove(queue)

    async def subscribe_to_task(task_id: TaskId) -> AsyncGenerator[EventUnion, None]:
        queue = asyncio.Queue()
        queues_by_task_id[task_id].append(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            queues_by_task_id[task_id].remove(queue)

    ###
    # Asynchronous API
    ###

    @app.post("/task", response_model=TaskId)
    async def create_task(
        parameters: ParamsUnion,
        task_service: TaskService = Depends(construct_task_service),
        user: User = Depends(get_user),
    ) -> TaskId:
        task = Task(
            parameters=parameters,
            user=user,
        )
        task_service.push_task(task)
        return task.task_id

    @app.get("/task/{task_id}", response_model=EventUnion)
    async def poll_task_status(
        task_id: TaskId,
        response: Response,
        status_service: StatusService = Depends(construct_status_service),
        user: User = Depends(get_user),
    ) -> EventUnion:
        response.headers["Cache-Control"] = "no-cache, no-store"  # don't cache poll requests

        task = status_service.get_task(task_id)
        if task is None or task.user.username != user.username:
            raise HTTPException(status_code=404, detail="Task not found")
        event = status_service.get_latest_event(task_id)
        if event is None:
            raise RuntimeError("Task exists but no event found")
        return event

    ###
    # Blobs (eventually to be replaced with a proper object store, and pre-signed POST/GET URLs)
    ###

    @app.get(
        "/blob/{blob_id}",
        responses={
            200: {
                "content": {"image/png": {}}
            },
            404: {
                "description": "Blob not found",
            }
        },
        # FIXME this throws an error, and I don't know why ;-;
        #  the only side effect is an erroneous "content-type: application/json" in the OpenAPI spec
        # response_model=Response
    )
    async def get_blob(
        blob_id: BlobId,
        blob_repo: BlobRepo = Depends(construct_blob_repo),
        user: User = Depends(get_user),
    ) -> Response:
        blob = blob_repo.get_blob(blob_id, user.username)
        if blob is None:
            raise HTTPException(status_code=404, detail="Blob not found")
        return Response(content=blob.data, media_type="image/png")

    @app.post("/blob", response_model=BlobId)
    async def post_blob(
        blob_data: UploadFile = File(),
        blob_repo: BlobRepo = Depends(construct_blob_repo),
        user: User = Depends(get_user),
    ) -> BlobId:
        blob = Blob(
            data=blob_data.file.read(),
            username=user.username,
        )
        return blob_repo.put_blob(blob)

    ###
    # Synchronous API (convenience wrappers for the asynchronous API)
    ###

    @app.get("/txt2img", responses={
        200: {
            "content": {
                "image/png": {},
            },
        },
    })
    async def txt2img(
        parameters: Txt2ImgParams = Depends(),
        user: User = Depends(get_user),
        task_service: TaskService = Depends(construct_task_service),
        blob_repo: BlobRepo = Depends(construct_blob_repo),
    ) -> Response:
        task = Task(
            parameters=parameters,
            user=user,
        )
        task_service.push_task(task)
        async for event in subscribe_to_task(task.task_id):
            if isinstance(event, CancelledEvent):
                raise HTTPException(status_code=500, detail=event.reason)
            if isinstance(event, FinishedEvent):
                blob_id = event.image.blob_id
                blob = blob_repo.get_blob(blob_id, username=user.username)
                if blob is None:
                    raise RuntimeError("Blob not found")
                return Response(content=blob.data, media_type="image/png")
        raise RuntimeError("Event stream ended unexpectedly")

    ###
    # Websocket
    ###

    async def get_ws_user(
        websocket: websockets.WebSocket,
        token: Union[str, None] = Query(default=None),
        user_repo: UserRepo = Depends(construct_user_repo),
    ) -> User:
        if token is None:
            await websocket.close(code=1008, reason="Missing token")
            raise HTTPException(status_code=401, detail="Unauthorized")

        try:
            return user_repo.must_get_user_by_token(token)
        except AuthenticationError:
            await websocket.close(code=1008, reason="Invalid token")
            raise HTTPException(status_code=401, detail="User not found")

    @app.websocket('/events')
    async def websocket_endpoint(
        websocket: websockets.WebSocket,
        user: User = Depends(get_ws_user),
    ):
        await websocket.accept()
        async for event in subscribe_to_session(user.session_id):
            await websocket.send_json(event.json())
        await websocket.close()

    ###
    # OpenAPI
    ###

    openapi_schema = app.openapi()
    with open('openapi.yml', 'w') as f:
        yaml.dump(openapi_schema, f)

    ###
    # Create default user
    ###

    def print_link_with_token() -> None:
        # construct token repo
        user_repo = app_config.user_repo(
            secret_key=app_config.SECRET_KEY,
            algorithm=app_config.ALGORITHM,
            access_token_expires=datetime.timedelta(minutes=app_config.ACCESS_TOKEN_EXPIRE_MINUTES),
            allow_public_token=app_config.ENABLE_PUBLIC_ACCESS,
        )

        # create default user
        username = "default_" + str(uuid.uuid4())
        password = bcrypt.hashpw(str(uuid.uuid4()).encode(), bcrypt.gensalt()).decode()
        user_repo.create_user(user=UserBase(username=username), password=password)
        token = user_repo.create_token_by_username_and_password(username=username, password=password)

        # print link
        print(f"Try visiting http://localhost:8000/txt2img?"
              f"prompt=corgi&"
              f"steps=2&"
              f"model_id=CompVis/stable-diffusion-v1-4&"
              f"token={token.access_token}")

    if app_config.PRINT_LINK_WITH_TOKEN:
        print_link_with_token()

    return app
