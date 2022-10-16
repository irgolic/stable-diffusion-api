from typing import Optional, Union, Type, Literal

import pydantic
from pydantic import conlist, Field

from stable_diffusion_server.models.model import ModelId


class Params(pydantic.BaseModel):
    model_id: ModelId = pydantic.Field(
        default="CompVis/stable-diffusion-v1-4",
        description="The model to use for image generation, e.g. 'CompVis/stable-diffusion-v1-4'.",
    )
    model_repository: Literal["huggingface"] = pydantic.Field(
        default="huggingface",
        description="The model repository to look up `model_id` in. "
                    "Currently only 'huggingface' is supported.",
    )
    prompt: str = pydantic.Field(
        description="The prompt to guide image generation."
    )
    negative_prompt: Optional[str] = pydantic.Field(
        default=None,
        description="The prompt to dissuade image generation. "
                    "Ignored when not using guidance (i.e., if `guidance` is `1`)."
    )
    steps: int = pydantic.Field(
        default=20,
        description="The number of denoising steps. "
                    "More denoising steps usually lead to a higher quality image at the expense of slower inference."
    )
    guidance: float = pydantic.Field(
        default=7.5,
        ge=1.0,
        description="Higher guidance encourages generation closely linked to `prompt`, "
                    "usually at the expense of lower image quality. "
                    "Try using more steps to improve image quality when using high guidance. "
                    "Guidance is disabled by setting `guidance` to `1`. "
                    "`guidance` is defined as `w` of equation 2. of "
                    "[ImagenPaper](https://arxiv.org/pdf/2205.11487.pdf). "
                    "See also: [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)."
    )
    num_images: int = pydantic.Field(
        default=1,
        description="The number of images to generate."
    )
    scheduler: Literal["plms", "ddim", "k-lms"] = pydantic.Field(
        default="plms",
        description="The scheduler to use for image generation. "
                    "Currently only 'plms', 'ddim', and 'k-lms', are supported."
    )
    # TODO implement seed
    # seed: Optional[int] = pydantic.Field(
    #     default=None,
    #     description="The randomness seed to use for image generation. "
    #                 "If not set, a random seed is used."
    # )


class Txt2ImgParams(Params):
    width: int = pydantic.Field(
        default=512,
        description="The pixel width of the generated image.")
    height: int = pydantic.Field(
        default=512,
        description="The pixel height of the generated image.")


class Img2ImgParams(Params):
    strength: float = pydantic.Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Conceptually, indicates how much to transform the image. "
                    "The image will be used as a starting point, adding more noise to it the larger the `strength`. "
                    "The number of denoising steps depends on the amount of noise initially added. "
                    "When `strength` is 1, it becomes pure noise, "
                    "and the denoising process will run for the full number of iterations specified in `steps`. "
                    "A value of 1, therefore, works like Txt2Img, essentially ignoring the reference image."
    )


ParamsUnion = Union[tuple(Params.__subclasses__())]  # type: ignore
