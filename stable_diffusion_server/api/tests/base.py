import asyncio
import io
import json
from typing import Any
from unittest import mock

import pytest
import pytest_asyncio
import requests

from stable_diffusion_server.api.tests.utils import AppClient


class BaseTestApp:
    @classmethod
    def get_client(cls) -> AppClient:
        raise NotImplementedError

    # fixtures

    @pytest.fixture(scope='function')
    def client(self):
        return self.get_client()

    @pytest_asyncio.fixture(scope="function")
    async def websocket(self, client):
        client.set_public_token()
        async with client.websocket_connect() as ws:
            yield ws
        ws.close()

    @pytest.fixture
    def dummy_txt2img_params(self):
        return {
            "task_type": "txt2img",
            "model_id": "hf-internal-testing/tiny-stable-diffusion-pipe",
            "prompt": "corgi wearing a top hat",
            "steps": 2,
            "safety_filter": False,
        }

    @pytest.fixture
    def resolved_dummy_txt2img_params(self):
        return {
            'task_type': 'txt2img',
            'model_id': 'hf-internal-testing/tiny-stable-diffusion-pipe',
            'model_provider': 'huggingface',
            'prompt': 'corgi wearing a top hat',
            'negative_prompt': None,
            'steps': 2,
            'guidance': 7.5,
            'scheduler': 'plms',
            'width': 512,
            'height': 512,
            "safety_filter": False,
        }

    @pytest.fixture
    def dummy_img2img_params(self, dummy_txt2img_params):
        return dummy_txt2img_params | {
            "task_type": "img2img",
            'initial_image': mock.ANY,
        }

    @pytest.fixture
    def resolved_dummy_img2img_params(self, dummy_img2img_params):
        return {
            'task_type': 'img2img',
            'model_id': 'hf-internal-testing/tiny-stable-diffusion-pipe',
            'model_provider': 'huggingface',
            'prompt': 'corgi wearing a top hat',
            'negative_prompt': None,
            'steps': 2,
            'guidance': 7.5,
            'scheduler': 'plms',
            "safety_filter": False,
            "strength": 0.8,
            'initial_image': mock.ANY,
        }

    # helper methods

    @staticmethod
    async def assert_websocket_received(event_dict: dict[str, Any], websocket) -> dict[str, Any]:
        # blocks indefinitely until the event is received
        ws_event = json.loads(json.loads(await websocket.recv()))
        assert ws_event == event_dict, ws_event
        return ws_event

    async def assert_poll_status(self, client, task_id, expected_event) -> dict[str, Any]:
        while True:
            response = client.get(f'/task/{task_id}')
            assert response.status_code == 200
            event = response.json()
            if event == expected_event:
                return event
            await asyncio.sleep(0.1)

    def post_task(self, client, params: dict[str, Any]) -> str:
        response = client.post('/task', json=params)
        assert response.status_code == 200
        return response.json()

    def get_blob(self, client, blob_id: str) -> requests.Response:
        response = client.get(f'/blob/{blob_id}')
        assert response.status_code == 200
        return response

    def post_blob(self, client, data: bytes) -> requests.Response:
        response = client.post('/blob', files={
            "blob_data": ("filename", io.BytesIO(data), "image/png"),
        })
        assert response.status_code == 200
        return response

    # tests

    @pytest.mark.asyncio
    async def test_txt2img_2img_with_token(
        self,
        client,
        websocket,  # token is automatically set in the websocket fixture
        dummy_txt2img_params,
        resolved_dummy_txt2img_params,
        dummy_img2img_params,
        resolved_dummy_img2img_params
    ):
        task_id = self.post_task(client, dummy_txt2img_params)

        # pending event
        await self.assert_websocket_received({
            'event_type': 'pending',
            'task_id': task_id,
        }, websocket)

        # started event
        await self.assert_websocket_received({
            'event_type': 'started',
            'task_id': task_id,
        }, websocket)

        # finished event
        expected_event = {
            'event_type': 'finished',
            'task_id': task_id,
            'image': {
                'blob_id': mock.ANY,
                'parameters_used': resolved_dummy_txt2img_params,
            }
        }

        ws_event = await self.assert_websocket_received(expected_event, websocket)
        poll_event = await self.assert_poll_status(client, task_id, expected_event)
        assert poll_event == ws_event

        generated_image_blob_id = ws_event['image']['blob_id']

        # download the blob
        response = self.get_blob(client, generated_image_blob_id)

        # upload the blob
        img_bytes = response.content
        uploaded_blob_id = self.post_blob(client, img_bytes).json()

        # run the generated image through img2img
        task_id = self.post_task(client,
            dummy_img2img_params | {
                "initial_image": uploaded_blob_id,
            }
        )

        # pending event
        await self.assert_websocket_received({
            'event_type': 'pending',
            'task_id': task_id,
        }, websocket)

        # started event
        await self.assert_websocket_received({
            'event_type': 'started',
            'task_id': task_id,
        }, websocket)

        # finished event
        expected_event = {
            'event_type': 'finished',
            'task_id': task_id,
            'image': {
                'blob_id': mock.ANY,
                'parameters_used': resolved_dummy_img2img_params | {
                    'initial_image': uploaded_blob_id,
                },
            }
        }

        ws_event = await self.assert_websocket_received(expected_event, websocket)
        poll_event = await self.assert_poll_status(client, task_id, expected_event)
        assert poll_event == ws_event

        generated_image_blob_id = ws_event['image']['blob_id']

        # download the blob (and do nothing with it)
        self.get_blob(client, generated_image_blob_id)

    @pytest.mark.asyncio
    async def test_txt2img_2img_without_token(
        self,
        client,
        dummy_txt2img_params,
        resolved_dummy_txt2img_params,
        dummy_img2img_params,
        resolved_dummy_img2img_params,
    ):
        task_id = self.post_task(client, dummy_txt2img_params)

        # finished event
        expected_event = {
            'event_type': 'finished',
            'task_id': task_id,
            'image': {
                'blob_id': mock.ANY,
                'parameters_used': resolved_dummy_txt2img_params,
            }
        }

        event = await self.assert_poll_status(client, task_id, expected_event)

        generated_image_blob_id = event['image']['blob_id']

        # download the blob
        response = self.get_blob(client, generated_image_blob_id)

        # upload the blob
        img_bytes = response.content
        uploaded_blob_id = self.post_blob(client, img_bytes).json()

        # run the generated image through img2img
        task_id = self.post_task(client,
            dummy_img2img_params | {
                "initial_image": uploaded_blob_id,
            }
        )

        # finished event
        expected_event = {
            'event_type': 'finished',
            'task_id': task_id,
            'image': {
                'blob_id': mock.ANY,
                'parameters_used': resolved_dummy_img2img_params | {
                    'initial_image': uploaded_blob_id,
                },
            }
        }

        event = await self.assert_poll_status(client, task_id, expected_event)

        generated_image_blob_id = event['image']['blob_id']

        # download the blob (and do nothing with it)
        self.get_blob(client, generated_image_blob_id)
