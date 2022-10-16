from typing import Union, Literal

import pydantic

from stable_diffusion_server.models.image import GeneratedImage
from stable_diffusion_server.models.task import TaskId
from stable_diffusion_server.models.user import SessionId


class Event(pydantic.BaseModel):
    event_type: str

    task_id: TaskId


class PendingEvent(Event):
    event_type: Literal['pending']


class StartedEvent(Event):
    event_type: Literal['started']


class CancelledEvent(Event):
    event_type: Literal['cancelled']

    reason: str


class FinishedEvent(Event):
    event_type: Literal["finished"]

    image: GeneratedImage


EventUnion = Union[tuple(Event.__subclasses__())]  # type: ignore
