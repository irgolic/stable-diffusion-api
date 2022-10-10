import asyncio
from typing import Coroutine

from stable_diffusion_server.models.task import Task


def get_runner_coroutine(task_listener, runner_service) -> Coroutine[Task, None, None]:
    async def runner_loop():
        async for task in task_listener.listen():
            await runner_service.run_task(task)

    return runner_loop()
