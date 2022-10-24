import uuid

import pydantic
from typing_extensions import TypeAlias

from stable_diffusion_server.models.params import ParamsUnion, AnyParams
from stable_diffusion_server.models.user import User

TaskId: TypeAlias = str


class Task(pydantic.BaseModel):
    parameters: AnyParams

    user: User
    task_id: TaskId = pydantic.Field(default_factory=lambda: TaskId(uuid.uuid4()))
