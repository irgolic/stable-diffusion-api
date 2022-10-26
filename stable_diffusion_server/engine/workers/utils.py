import asyncio
import os
from typing import Coroutine

from stable_diffusion_server.models.task import Task


def get_runner_coroutine(task_listener, runner_service) -> Coroutine[Task, None, None]:
    async def runner_loop():
        try:
            async for task in task_listener.listen():
                await runner_service.run_task(task)
        except Exception as e:
            print(f"Error running task: {e}")

    return runner_loop()


def get_local_blob_repo_params():
    # TODO move away from hosting blobs alongside the API
    #  using an external image storage service is preferable, helps to avoid duplicating
    #  the base blob url and encryption parameters in the api AND each runner for token generation
    base_url = os.environ.get("BASE_URL", "http://localhost:8000")
    return dict(
        base_blob_url=base_url + "/blob",
        secret_key=os.environ["SECRET_KEY"],
        algorithm="HS256",
    )
