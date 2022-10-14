import asyncio

from stable_diffusion_server.api.base import AppConfig, create_app
from stable_diffusion_server.engine.repos.blob_repo import InMemoryBlobRepo
from stable_diffusion_server.engine.repos.key_value_repo import InMemoryKeyValueRepo
from stable_diffusion_server.engine.repos.messaging_repo import InMemoryMessagingRepo
from stable_diffusion_server.engine.repos.user_repo import InMemoryUserRepo
from stable_diffusion_server.engine.workers.in_memory_worker import create_runner

app_config = AppConfig(
    blob_repo_class=InMemoryBlobRepo,
    messaging_repo_class=InMemoryMessagingRepo,
    user_repo=InMemoryUserRepo,
    key_value_repo=InMemoryKeyValueRepo,
)

fastapi_app = create_app(app_config)


def app(scope, receive, send):
    loop = asyncio.get_event_loop()
    loop.create_task(create_runner())
    return fastapi_app(scope, receive, send)
