import json
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
                "model_id": "dummy",
                "prompt": [
                    {
                        "text": "dummy",
                    }
                ]
            }
            response = self.client.post('/txt2img', json=params)
            assert response.status_code == 200
            task_id = response.json()
            ws_event = await ws.recv()
            assert json.loads(json.loads(ws_event)) == {
                'event_type': 'finished',
                'task_id': task_id,
                'image': {
                    'link': 'dummy',
                    'format': 'dummy',
                    'parameters_used': {
                        'model_id': 'dummy',
                        'prompt': [
                            {
                                'text': 'dummy',
                                'alt_text': None,
                                'emphasis': 0,
                                'percentage_divider': None,
                            },
                        ],
                        'negative_prompt': None,
                        'step_count': 20,
                        'width': 512,
                        'height': 512,
                    },
                }
            }
