import asyncio
import logging
import sys
from typing import Coroutine

from stable_diffusion_server.engine.repos.blob_repo import RedisBlobRepo
from stable_diffusion_server.engine.repos.messaging_repo import RedisMessagingRepo
from stable_diffusion_server.engine.services.event_service import EventService
from stable_diffusion_server.engine.services.runner_service import RunnerService
from stable_diffusion_server.engine.services.task_service import TaskListener
from stable_diffusion_server.engine.workers.utils import get_runner_coroutine
from stable_diffusion_server.models.task import Task

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)


def create_runner() -> Coroutine[Task, None, None]:
    logger.info("Starting redis worker")

    # instantiate redis messaging repo
    messaging_repo = RedisMessagingRepo()

    # instantiate runner service
    blob_repo = RedisBlobRepo()
    event_service = EventService(
        messaging_repo=messaging_repo
    )
    runner_service = RunnerService(
        blob_repo=blob_repo,
        event_service=event_service,
    )

    # listen for tasks
    task_listener = TaskListener(
        messaging_repo=messaging_repo,
    )

    return get_runner_coroutine(task_listener, runner_service)


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(create_runner())
