import logging
from typing import AsyncIterator

import pydantic

from stable_diffusion_server.engine.repos.messaging_repo import MessagingRepo
from stable_diffusion_server.models.task import TaskUnion

logger = logging.getLogger(__name__)


class TaskService:
    def __init__(
        self,
        messaging_repo: MessagingRepo,
    ):
        self.messaging_repo = messaging_repo

    def push_task(self, task: TaskUnion) -> None:
        self.messaging_repo.push('task', task.json())


class TaskListener:
    def __init__(
        self,
        messaging_repo: MessagingRepo,
    ):
        self.messaging_repo = messaging_repo

    async def get_task(self) -> TaskUnion:
        task_json = await self.messaging_repo.pop('task')
        return pydantic.parse_raw_as(TaskUnion, task_json)

    async def listen(self) -> AsyncIterator[TaskUnion]:
        while True:
            yield await self.get_task()
