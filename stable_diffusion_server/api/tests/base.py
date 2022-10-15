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
    async def test_dummy(self):
        async with self.client.websocket_connect() as ws:
            params = {
                "prompt": "corgi wearing a top hat",
                "steps": 2,
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
                    'format': 'png',
                    'parameters_used': {
                        'model_id': 'CompVis/stable-diffusion-v1-4',
                        'model_repository': 'huggingface',
                        'prompt': 'corgi wearing a top hat',
                        'negative_prompt': None,
                        'steps': 2,
                        'guidance': 7.5,
                        'num_images': 1,
                        'seed': None,
                        'scheduler': 'plms',
                        'width': 512,
                        'height': 512,
                    },
                }
            }

            ws_event = json.loads(json.loads(await ws.recv()))
            assert ws_event == expected_event

            # poll status
            response = self.client.get(f'/task/{task_id}')
            assert response.status_code == 200
            assert response.json() == expected_event
