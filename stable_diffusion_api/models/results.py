import pydantic

from stable_diffusion_api.models.blob import BlobUrl
from stable_diffusion_api.models.params import ParamsUnion


class GeneratedBlob(pydantic.BaseModel):
    blob_url: BlobUrl
    parameters_used: ParamsUnion
