import asyncio
import logging
import os
import sys
from typing import Coroutine

from stable_diffusion_api.engine.repos.blob_repo import InMemoryBlobRepo
from stable_diffusion_api.engine.repos.key_value_repo import InMemoryKeyValueRepo
from stable_diffusion_api.engine.repos.messaging_repo import InMemoryMessagingRepo
from stable_diffusion_api.engine.services.event_service import EventService
from stable_diffusion_api.engine.services.runner_service import RunnerService
from stable_diffusion_api.engine.services.status_service import StatusService
from stable_diffusion_api.engine.services.task_service import TaskListener
from stable_diffusion_api.engine.workers.utils import get_runner_coroutine, get_local_blob_repo_params
from stable_diffusion_api.models.task import Task

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)


def create_runner() -> Coroutine[Task, None, None]:
    logger.info("Starting in memory worker")

    # instantiate redis messaging repo
    messaging_repo = InMemoryMessagingRepo()

    # instantiate runner service
    blob_repo = InMemoryBlobRepo(**get_local_blob_repo_params())
    key_value_repo = InMemoryKeyValueRepo()
    status_service = StatusService(
        key_value_repo=key_value_repo,
    )
    event_service = EventService(
        messaging_repo=messaging_repo,
        status_service=status_service,
    )
    runner_service = RunnerService(
        blob_repo=blob_repo,
        status_service=status_service,
        event_service=event_service,
    )

    # listen for tasks
    task_listener = TaskListener(
        messaging_repo=messaging_repo,
    )

    return get_runner_coroutine(task_listener, runner_service)
