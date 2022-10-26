import asyncio
import io
import json
import urllib.parse
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

    @pytest_asyncio.fixture(scope="function")
    async def client(self):
        async with self.get_client() as c:
            yield c

    @pytest_asyncio.fixture(scope="function")
    async def websocket(self, client):
        await client.set_public_token()
        async with client.websocket_connect() as ws:
            yield ws
        ws.close()

    @pytest.fixture
    def common_params(self):
        return {
            "model": "hf-internal-testing/tiny-stable-diffusion-pipe",
            "prompt": "corgi wearing a top hat",
            "steps": 2,
            "safety_filter": False,
        }

    @pytest.fixture
    def resolved_common_params(self, common_params):
        return common_params | {
            'negative_prompt': None,
            'guidance': 7.5,
            'scheduler': 'plms',
            'seed': mock.ANY,
        }

    @pytest.fixture
    def dummy_txt2img_params(self, common_params):
        return common_params | {
            "params_type": "txt2img",
        }

    @pytest.fixture
    def resolved_dummy_txt2img_params(self, resolved_common_params):
        return resolved_common_params | {
            "params_type": "txt2img",
            'width': 512,
            'height': 512,
        }

    @pytest.fixture
    def dummy_img2img_params(self, common_params):
        return common_params | {
            "params_type": "img2img",
            'initial_image': mock.ANY,
        }

    @pytest.fixture
    def resolved_dummy_img2img_params(self, resolved_common_params):
        return resolved_common_params | {
            "params_type": "img2img",
            'initial_image': mock.ANY,
            "strength": 0.8,
        }

    @pytest.fixture
    def dummy_inpaint_params(self, common_params):
        return common_params | {
            "params_type": "inpaint",
            'initial_image': mock.ANY,
            "mask": mock.ANY,
        }

    @pytest.fixture
    def resolved_dummy_inpaint_params(self, resolved_common_params):
        return resolved_common_params | {
            "params_type": "inpaint",
            'initial_image': mock.ANY,
            "mask": mock.ANY,
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
            response = await client.get(f'/task/{task_id}')
            assert response.status_code == 200
            event = response.json()
            if event == expected_event:
                return event
            await asyncio.sleep(0.1)

    async def post_task(
        self,
        client,
        params: dict[str, Any],
        resolved_params: dict[str, Any],
        websocket=None
    ) -> dict[str, Any]:
        response = await client.post('/task', json=params)
        assert response.status_code == 200
        task_id = response.json()

        expected_event = {
            'event_type': 'finished',
            'task_id': task_id,
            'image': {
                'blob_url': mock.ANY,
                'parameters_used': resolved_params,
            }
        }

        if websocket is not None:
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
            await self.assert_websocket_received(expected_event, websocket)

        poll_event = await self.assert_poll_status(client, task_id, expected_event)
        return poll_event

    async def get_blob(self, client, blob_url: str) -> requests.Response:
        path = urllib.parse.urlparse(blob_url).path
        response = await client.get(path)
        assert response.status_code == 200
        return response

    async def post_blob(self, client, data: bytes) -> requests.Response:
        response = await client.post('/blob', files={
            "blob_data": ("filename", io.BytesIO(data), "image/png"),
        })
        assert response.status_code == 200
        return response

    # tests

    @pytest.mark.asyncio
    async def test_txt2img_2img_inpaint_with_token(
        self,
        client,
        websocket,  # token is automatically set in the websocket fixture
        dummy_txt2img_params,
        resolved_dummy_txt2img_params,
        dummy_img2img_params,
        resolved_dummy_img2img_params,
        dummy_inpaint_params,
        resolved_dummy_inpaint_params,
    ):
        finished_event = await self.post_task(
            client,
            dummy_txt2img_params,
            resolved_dummy_txt2img_params,
            websocket=websocket,
        )

        # assert seed got set after randomization
        assert finished_event['image']['parameters_used']['seed'] is not None

        generated_image_blob_url = finished_event['image']['blob_url']

        # download the blob
        response = await self.get_blob(client, generated_image_blob_url)

        # upload the blob
        img_bytes = response.content
        upload_response = await self.post_blob(client, img_bytes)
        uploaded_blob_url = upload_response.json()

        # use manual seed
        manual_seed = 42

        # run the generated image through img2img
        finished_event = await self.post_task(
            client,
            dummy_img2img_params | {
                "initial_image": uploaded_blob_url,
                'seed': manual_seed,
            },
            resolved_dummy_img2img_params | {
                "initial_image": uploaded_blob_url,
                'seed': manual_seed,
            },
            websocket,
        )

        # download the blob (and do nothing with it)
        generated_image_blob_url = finished_event['image']['blob_url']
        await self.get_blob(client, generated_image_blob_url)

        # run the generated image with the previous image as mask through inpaint
        finished_event = await self.post_task(
            client,
            dummy_inpaint_params | {
                "initial_image": uploaded_blob_url,
                "mask": generated_image_blob_url,
            },
            resolved_dummy_inpaint_params | {
                "initial_image": uploaded_blob_url,
                "mask": generated_image_blob_url,
            },
            websocket,
        )

        # download the blob (and do nothing with it)
        generated_image_blob_url = finished_event['image']['blob_url']
        await self.get_blob(client, generated_image_blob_url)

    @pytest.mark.asyncio
    async def test_txt2img_2img_without_token(
        self,
        client,
        dummy_txt2img_params,
        resolved_dummy_txt2img_params,
        dummy_img2img_params,
        resolved_dummy_img2img_params,
    ):
        finished_event = await self.post_task(client, dummy_txt2img_params, resolved_dummy_txt2img_params)

        generated_image_blob_url = finished_event['image']['blob_url']

        # download the blob
        response = await self.get_blob(client, generated_image_blob_url)

        # upload the blob
        img_bytes = response.content
        upload_response = await self.post_blob(client, img_bytes)
        uploaded_blob_url = upload_response.json()

        # use manual seed
        manual_seed = 42

        # run the generated image through img2img
        finished_event = await self.post_task(
            client,
            dummy_img2img_params | {
                "initial_image": uploaded_blob_url,
                'seed': manual_seed,
            },
            resolved_dummy_img2img_params | {
                "initial_image": uploaded_blob_url,
                'seed': manual_seed,
            },
        )

        generated_image_blob_url = finished_event['image']['blob_url']

        # download the blob (and do nothing with it)
        await self.get_blob(client, generated_image_blob_url)

    @pytest.mark.asyncio
    async def test_sync_txt2img(
        self,
        client,
        dummy_txt2img_params,
        resolved_dummy_txt2img_params,
    ):
        response = await client.get('/txt2img', params=dummy_txt2img_params)
        assert response.status_code == 200
        assert response.headers['Content-Type'] == 'application/json'

        generated_image = response.json()
        assert generated_image['parameters_used'] == resolved_dummy_txt2img_params
