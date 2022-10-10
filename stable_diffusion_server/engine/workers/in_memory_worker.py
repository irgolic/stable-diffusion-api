import asyncio
import logging
import sys
from typing import Coroutine

from stable_diffusion_server.engine.repos.blob_repo import InMemoryBlobRepo
from stable_diffusion_server.engine.repos.messaging_repo import InMemoryMessagingRepo
from stable_diffusion_server.engine.services.event_service import EventService
from stable_diffusion_server.engine.services.runner_service import RunnerService
from stable_diffusion_server.engine.services.task_service import TaskListener
from stable_diffusion_server.engine.workers.utils import get_runner_coroutine
from stable_diffusion_server.models.task import Task

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)


def create_runner() -> Coroutine[Task, None, None]:
    logger.info("Starting in memory worker")

    # instantiate redis messaging repo
    messaging_repo = InMemoryMessagingRepo()

    # instantiate runner service
    blob_repo = InMemoryBlobRepo()
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