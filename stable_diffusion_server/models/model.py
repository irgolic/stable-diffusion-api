import pydantic
from typing_extensions import TypeAlias

ModelId: TypeAlias = str


class Model(pydantic.BaseModel):
    id: ModelId
    name: str
    description: str
    author: str
