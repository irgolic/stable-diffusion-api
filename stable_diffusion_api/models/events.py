from typing import Union, Literal

import pydantic

from stable_diffusion_api.models.results import GeneratedBlob
from stable_diffusion_api.models.task import TaskId
from stable_diffusion_api.models.user import SessionId


class Event(pydantic.BaseModel):
    event_type: str

    task_id: TaskId


class PendingEvent(Event):
    event_type: Literal['pending']


class StartedEvent(Event):
    event_type: Literal['started']


class AbortedEvent(Event):
    event_type: Literal['aborted']

    reason: str


class FinishedEvent(Event):
    event_type: Literal["finished"]

    result: GeneratedBlob


EventUnion = Union[tuple(Event.__subclasses__())]  # type: ignore
