import asyncio
import os
from unittest import mock

import pytest
from httpx import AsyncClient
from requests_toolbelt import sessions

from stable_diffusion_api.api.tests.base import BaseTestApp
from stable_diffusion_api.api.tests.utils import RemoteAppClient


@pytest.mark.e2e
class TestRedisAppE2E(BaseTestApp):
    @classmethod
    def get_client(cls):
        return RemoteAppClient(
            # sessions.BaseUrlSession(base_url=os.environ.get('API_URL', 'http://127.0.0.1:8000'))
            AsyncClient(base_url=os.environ.get('API_URL', 'http://127.0.0.1:8000')),
        )

    @pytest.mark.asyncio
    async def test_cancel_sync_txt2img(
        self,
        client,
        websocket,
        dummy_txt2img_params,
    ):
        coro = client.get('/txt2img', params=dummy_txt2img_params)
        task = asyncio.create_task(coro)
        await asyncio.sleep(0.1)
        task.cancel()

        # check events
        await self.assert_websocket_received({
            'event_type': 'pending',
            'task_id': mock.ANY,
        }, websocket)

        await self.assert_websocket_received({
            'event_type': 'started',
            'task_id': mock.ANY,
        }, websocket)

        await self.assert_websocket_received({
            'event_type': 'aborted',
            'task_id': mock.ANY,
            'reason': 'Task cancelled by user',
        }, websocket)
