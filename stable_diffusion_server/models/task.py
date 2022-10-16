import uuid
from typing import Union, Literal

import pydantic
from typing_extensions import TypeAlias

from stable_diffusion_server.models.params import Txt2ImgParams, Img2ImgParams, Params
from stable_diffusion_server.models.user import User

TaskId: TypeAlias = str


class Task(pydantic.BaseModel):
    task_type: str
    parameters: Params

    user: User
    task_id: TaskId = pydantic.Field(default_factory=lambda: TaskId(uuid.uuid4()))


class Txt2ImgTask(Task):
    task_type: Literal["txt2img"]
    parameters: Txt2ImgParams


class Img2ImgTask(Task):
    task_type: Literal["img2img"]
    parameters: Img2ImgParams


# class InpaintingTask(Task):
#     task_type: Literal["inpainting"]
#
#     parameters: Params
#     input_image: Image
#     input_mask: Image


TaskUnion = Union[tuple(Task.__subclasses__())]  # type: ignore
