import pydantic

from stable_diffusion_server.engine.repos.blob_repo import BlobId
from stable_diffusion_server.models.params import ParamsUnion


class Image(pydantic.BaseModel):
    blob_id: BlobId
    format: str


class GeneratedImage(Image):
    parameters_used: ParamsUnion
    link: str
