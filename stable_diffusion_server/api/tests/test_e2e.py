import os

import pytest
from requests_toolbelt import sessions

from stable_diffusion_server.api.tests.base import BaseTestApp
from stable_diffusion_server.api.tests.utils import RemoteAppClient


@pytest.mark.e2e
class TestRedisAppE2E(BaseTestApp):
    @classmethod
    def get_client(cls):
        return RemoteAppClient(
            sessions.BaseUrlSession(base_url=os.environ.get('API_URL', 'http://127.0.0.1:8000'))
        )
