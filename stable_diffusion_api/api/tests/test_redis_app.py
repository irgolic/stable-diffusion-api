import asyncio
import logging

import pytest
import redis
from asgi_lifespan import LifespanManager
from httpx import AsyncClient

from stable_diffusion_api.api import redis_app
from stable_diffusion_api.api.tests.base import BaseTestApp
from stable_diffusion_api.api.tests.utils import AsyncioTestClient, LocalAppClient
from stable_diffusion_api.engine.utils import load_redis
from stable_diffusion_api.engine.workers.redis_worker import create_runner

logger = logging.getLogger(__name__)

try:
    r = load_redis()
    r.ping()
except redis.exceptions.RedisError as e:
    logger.error(f"Redis is not available: {e}")
    pytest.skip(f"Could not connect to redis instance, skipping redis api tests", allow_module_level=True)


class TestLiveRedisApp(BaseTestApp):
    _runner_task = None

    @classmethod
    def get_client(cls):
        loop = asyncio.get_event_loop()
        if cls._runner_task is None:
            cls._runner_task = loop.create_task(create_runner())
        return LocalAppClient(
            AsyncClient(app=redis_app.app, base_url="http://testserver"),
            AsyncioTestClient(event_loop=loop,
                              app=redis_app.app),
            LifespanManager(redis_app.app),
        )
