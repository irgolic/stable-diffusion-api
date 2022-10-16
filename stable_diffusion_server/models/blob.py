import pydantic
from typing_extensions import TypeAlias

from stable_diffusion_server.models.user import Username

BlobId: TypeAlias = str


class Blob(pydantic.BaseModel):
    data: bytes
    username: Username
