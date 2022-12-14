import asyncio

import pytest
from httpx import AsyncClient
from asgi_lifespan import LifespanManager

from stable_diffusion_api.api.tests.base import BaseTestApp

from stable_diffusion_api.api.tests.utils import AsyncioTestClient, LocalAppClient
from stable_diffusion_api.api.in_memory_app import fastapi_app
from stable_diffusion_api.engine.workers.in_memory_worker import create_runner


class TestInMemoryApp(BaseTestApp):
    _runner_task = None

    @classmethod
    def get_client(cls):
        loop = asyncio.get_event_loop()
        if cls._runner_task is None:
            cls._runner_task = loop.create_task(create_runner())
        return LocalAppClient(
            AsyncClient(app=fastapi_app, base_url="http://testserver"),
            AsyncioTestClient(event_loop=loop,
                              app=fastapi_app),
            LifespanManager(fastapi_app),
        )
