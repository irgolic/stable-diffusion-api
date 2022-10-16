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


_runner_task = None


async def app(scope, receive, send):
    global _runner_task
    if _runner_task is None:
        loop = asyncio.get_event_loop()
        _runner_task = loop.create_task(create_runner())
    await fastapi_app(scope, receive, send)
