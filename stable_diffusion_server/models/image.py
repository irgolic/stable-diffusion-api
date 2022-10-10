import pydantic

from stable_diffusion_server.models.params import Params


class Image(pydantic.BaseModel):
    link: str
    format: str


class GeneratedImage(Image):
    parameters_used: Params
