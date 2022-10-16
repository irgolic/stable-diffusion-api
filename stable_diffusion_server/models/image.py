import pydantic

from stable_diffusion_server.engine.repos.blob_repo import BlobId
from stable_diffusion_server.models.params import ParamsUnion


class GeneratedImage(pydantic.BaseModel):
    blob_id: BlobId
    parameters_used: ParamsUnion
    link: str
