import uuid

import pydantic
from typing_extensions import TypeAlias

from stable_diffusion_api.models.params import ParamsUnion
from stable_diffusion_api.models.user import User

TaskId: TypeAlias = str


class Task(pydantic.BaseModel):
    parameters: ParamsUnion

    user: User
    task_id: TaskId = pydantic.Field(default_factory=lambda: TaskId(uuid.uuid4()))
