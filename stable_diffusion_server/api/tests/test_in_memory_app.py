import asyncio

from starlette.testclient import TestClient

from stable_diffusion_server.api.tests.base import BaseTestApp

from stable_diffusion_server.api.tests.utils import AsyncioTestClient, LocalAppClient
from stable_diffusion_server.api.in_memory_app import fastapi_app
from stable_diffusion_server.engine.workers.in_memory_worker import create_runner


class TestInMemoryApp(BaseTestApp):
    @classmethod
    def get_client(cls):
        loop = asyncio.get_event_loop()
        loop.create_task(create_runner())
        return LocalAppClient(
            TestClient(fastapi_app),
            AsyncioTestClient(event_loop=loop,
                              app=fastapi_app)
        )
