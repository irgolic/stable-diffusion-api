import json
import typing
from unittest import mock

import pytest

from stable_diffusion_server.api.tests.utils import AppClient


class BaseTestApp:
    @classmethod
    def get_client(cls) -> AppClient:
        raise NotImplementedError

    @classmethod
    def setup_class(cls):
        cls.client = cls.get_client()

    @pytest.mark.asyncio
    async def test_txt2img_2img(self):
        async with self.client.websocket_connect() as ws:
            params = {
                "model_id": "hf-internal-testing/tiny-stable-diffusion-pipe",
                "prompt": "corgi wearing a top hat",
                "steps": 2,
                "safety_filter": False,
            }
            response = self.client.post('/txt2img', json=params)
            assert response.status_code == 200
            task_id = response.json()

            # pending event
            ws_event = json.loads(json.loads(await ws.recv()))
            assert ws_event == {
                'event_type': 'pending',
                'task_id': task_id,
            }

            # started event
            ws_event = json.loads(json.loads(await ws.recv()))
            assert ws_event == {
                'event_type': 'started',
                'task_id': task_id,
            }

            # finished event
            expected_event = {
                'event_type': 'finished',
                'task_id': task_id,
                'image': {
                    'blob_id': mock.ANY,
                    'link': mock.ANY,
                    'parameters_used': {
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
                    },
                }
            }

            ws_event = json.loads(json.loads(await ws.recv()))
            assert ws_event == expected_event

            # poll status
            response = self.client.get(f'/task/{task_id}')
            assert response.status_code == 200
            assert response.json() == expected_event

            generated_image_blob_id = ws_event['image']['blob_id']

            # run the generated image through img2img
            response = self.client.post(
                '/img2img',
                json={
                    "initial_image": generated_image_blob_id,
                } | params
            )
            assert response.status_code == 200
            task_id = response.json()

            # pending event
            ws_event = json.loads(json.loads(await ws.recv()))
            assert ws_event == {
                'event_type': 'pending',
                'task_id': task_id,
            }

            # started event
            ws_event = json.loads(json.loads(await ws.recv()))
            assert ws_event == {
                'event_type': 'started',
                'task_id': task_id,
            }

            # finished event
            expected_event = {
                'event_type': 'finished',
                'task_id': task_id,
                'image': {
                    'blob_id': mock.ANY,
                    'link': mock.ANY,
                    'parameters_used': {
                        'task_type': 'img2img',
                        'model_id': 'hf-internal-testing/tiny-stable-diffusion-pipe',
                        'model_provider': 'huggingface',
                        'prompt': 'corgi wearing a top hat',
                        'negative_prompt': None,
                        'steps': 2,
                        'guidance': 7.5,
                        'scheduler': 'plms',
                        'initial_image': generated_image_blob_id,
                        "safety_filter": False,
                        "strength": 0.8,
                    },
                }
            }

            ws_event = json.loads(json.loads(await ws.recv()))
            assert ws_event == expected_event
