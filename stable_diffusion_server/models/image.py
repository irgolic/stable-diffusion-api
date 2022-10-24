import pydantic

from stable_diffusion_server.models.blob import BlobUrl
from stable_diffusion_server.models.params import ParamsUnion, AnyParams


class GeneratedImage(pydantic.BaseModel):
    image_url: BlobUrl
    parameters_used: AnyParams
