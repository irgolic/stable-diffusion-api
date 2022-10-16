import pydantic

from stable_diffusion_server.models.blob import BlobId
from stable_diffusion_server.models.params import ParamsUnion


class GeneratedImage(pydantic.BaseModel):
    blob_id: BlobId
    parameters_used: ParamsUnion
