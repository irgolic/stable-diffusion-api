import logging
from typing import AsyncIterator

import pydantic

from stable_diffusion_api.engine.repos.messaging_repo import MessagingRepo
from stable_diffusion_api.engine.services.event_service import EventService
from stable_diffusion_api.engine.services.status_service import StatusService
from stable_diffusion_api.models.events import PendingEvent
from stable_diffusion_api.models.task import Task

logger = logging.getLogger(__name__)


class TaskService:
    def __init__(
        self,
        messaging_repo: MessagingRepo,
        event_service: EventService,
        status_service: StatusService,
    ):
        self.messaging_repo = messaging_repo
        self.event_service = event_service
        self.status_service = status_service

    def push_task(self, task: Task) -> None:
        # register task
        self.status_service.store_task(task)
        # advertise task pending status
        self.event_service.send_event(
            task.user.session_id,
            PendingEvent(
                event_type="pending",
                task_id=task.task_id,
            )
        )
        # push task
        self.messaging_repo.push('task_queue', task.json())


class TaskListener:
    def __init__(
        self,
        messaging_repo: MessagingRepo,
    ):
        self.messaging_repo = messaging_repo

    async def get_task(self) -> Task:
        task_json = await self.messaging_repo.pop('task_queue')
        return Task.parse_raw(task_json)

    async def listen(self) -> AsyncIterator[Task]:
        while True:
            yield await self.get_task()
