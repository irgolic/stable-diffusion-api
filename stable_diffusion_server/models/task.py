import uuid
from typing import Union, Literal

import pydantic
from typing_extensions import TypeAlias

from stable_diffusion_server.models.image import Image
from stable_diffusion_server.models.params import Params
from stable_diffusion_server.models.user import SessionId

TaskId: TypeAlias = str


class Task(pydantic.BaseModel):
    task_type: str

    session_id: SessionId
    task_id: TaskId = pydantic.Field(default_factory=lambda: TaskId(uuid.uuid4()))


class Txt2ImgTask(Task):
    task_type: Literal["txt2img"]

    params: Params


class Img2ImgTask(Task):
    task_type: Literal["img2img"]

    params: Params
    image: Image


TaskUnion = Union[tuple(Task.__subclasses__())]  # type: ignore
