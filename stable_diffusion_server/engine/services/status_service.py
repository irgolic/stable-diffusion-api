from typing import Optional

import pydantic

from stable_diffusion_server.engine.repos.key_value_repo import KeyValueRepo
from stable_diffusion_server.models.events import EventUnion
from stable_diffusion_server.models.task import TaskId, Task


class StatusService:
    def __init__(
        self,
        key_value_repo: KeyValueRepo
    ):
        self.key_value_repo = key_value_repo

    def store_task(self, task: Task) -> None:
        self.key_value_repo.store('task', task.task_id, task.json())

    def get_task(self, task_id: TaskId) -> Optional[Task]:
        task_json = self.key_value_repo.retrieve('task', task_id)
        if task_json is None:
            return None
        return pydantic.parse_raw_as(Task, task_json)

    def store_event(self, event: EventUnion) -> None:
        if not self.key_value_repo.exists('task', event.task_id):
            raise ValueError(f"Task {event.task_id} does not exist")
        # store latest task event
        self.key_value_repo.store('task_event', event.task_id, event.json())

    def get_latest_event(self, task_id: TaskId) -> Optional[EventUnion]:
        # retrieve latest task event
        event_json = self.key_value_repo.retrieve('task_event', task_id)
        if event_json is None:
            return None
        return pydantic.parse_raw_as(EventUnion, event_json)
