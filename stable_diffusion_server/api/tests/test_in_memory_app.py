import asyncio

import pytest
from httpx import AsyncClient
from starlette.testclient import TestClient

from stable_diffusion_server.api.tests.base import BaseTestApp

from stable_diffusion_server.api.tests.utils import AsyncioTestClient, LocalAppClient
from stable_diffusion_server.api.in_memory_app import fastapi_app
from stable_diffusion_server.engine.workers.in_memory_worker import create_runner


class TestInMemoryApp(BaseTestApp):
    _runner_task = None

    @classmethod
    def get_client(cls):
        loop = asyncio.get_event_loop()
        # pytest starts a new event loop for each test, so we need to create a new runner for each test
        if cls._runner_task is None or cls._runner_task.get_loop() is not loop:
            cls._runner_task = loop.create_task(create_runner())
        return LocalAppClient(
            AsyncClient(app=fastapi_app, base_url="http://testserver"),
            AsyncioTestClient(event_loop=loop,
                              app=fastapi_app)
        )
