import pydantic

from stable_diffusion_server.models.blob import BlobUrl
from stable_diffusion_server.models.params import ParamsUnion


class GeneratedBlob(pydantic.BaseModel):
    image_url: BlobUrl
    parameters_used: ParamsUnion
