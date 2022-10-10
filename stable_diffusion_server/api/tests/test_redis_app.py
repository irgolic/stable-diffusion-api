import asyncio

import pytest
import redis
from starlette.testclient import TestClient

from stable_diffusion_server.api import redis_app
from stable_diffusion_server.api.tests.base import BaseTestApp
from stable_diffusion_server.api.tests.utils import AsyncioTestClient, LocalAppClient
from stable_diffusion_server.engine.utils import load_redis
from stable_diffusion_server.engine.workers.redis_worker import create_runner

try:
    r = load_redis()
    r.ping()
except redis.exceptions.RedisError:
    pytest.skip("Could not connect to redis instance, skipping redis api tests", allow_module_level=True)


class TestLiveRedisApp(BaseTestApp):
    @classmethod
    def get_client(cls):
        loop = asyncio.get_event_loop()
        loop.create_task(create_runner())
        return LocalAppClient(
            TestClient(redis_app.app),
            AsyncioTestClient(event_loop=loop,
                              app=redis_app.app)
        )
