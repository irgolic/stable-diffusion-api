from typing import Optional, Union, Literal

import pydantic
import typing

from stable_diffusion_server.models.blob import BlobUrl


class Params(pydantic.BaseModel):
    _endpoint_stem: typing.ClassVar[str]

    class Config:
        extra = pydantic.Extra.forbid

    pipeline: str = pydantic.Field(
        description="The pipeline to use for the task. "
                    "One of: "
                    "the *file name* of a community pipeline hosted on GitHub under "
                    "https://github.com/huggingface/diffusers/tree/main/examples/community "
                    "(e.g., 'clip_guided_stable_diffusion'), "
                    "*a path* to a *directory* containing a file called `pipeline.py` "
                    "(e.g., './my_pipeline_directory/'.), or "
                    "the *repo id* of a custom pipeline hosted on huggingface. "
                    "See [Loading and Creating Custom Pipelines]"
                    "(https://huggingface.co/docs/diffusers/main/en/using-diffusers/custom_pipelines). "
    )
    pipeline_method: Optional[str] = pydantic.Field(
        description="The method to call on the pipeline. "
                    "If unspecified, the pipeline itself will be called.",
    )

    model: str = pydantic.Field(
        default="CompVis/stable-diffusion-v1-4",
        description="The model to use for image generation. "
                    "One of: "
                    "the *repo id* of a pretrained pipeline hosted on huggingface "
                    "(e.g. 'CompVis/stable-diffusion-v1-4'), "
                    "*a path* to a *directory* containing pipeline weights, "
                    "(e.g., './my_model_directory/'). ",
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
    scheduler: Literal["plms", "ddim", "k-lms"] = pydantic.Field(
        default="plms",
        description="The scheduler to use for image generation. "
                    "Currently only 'plms', 'ddim', and 'k-lms', are supported."
    )
    safety_filter: bool = pydantic.Field(
        default=True,
        description="Ensure that you abide by the conditions of the Stable Diffusion license and "
                    "do not expose unfiltered results in services or applications open to the public. "
                    "For more information, please see https://github.com/huggingface/diffusers/pull/254",
    )
    seed: Optional[int] = pydantic.Field(
        default=None,
        description="The randomness seed to use for image generation. "
                    "If not set, a random seed is used."
    )
    extra_parameters: Optional[typing.Dict[str, typing.Any]] = pydantic.Field(
        default={},
        description="Extra parameters to pass to the pipeline. "
                    "See the documentation of the pipeline for more information."
    )


class Txt2ImgParams(Params):
    _endpoint_stem = 'txt2img'

    pipeline: Literal["stable_diffusion_mega"] = "stable_diffusion_mega"
    pipeline_method: Literal["text2img"] = "text2img"

    width: int = pydantic.Field(
        default=512,
        description="The pixel width of the generated image.")
    height: int = pydantic.Field(
        default=512,
        description="The pixel height of the generated image.")


class Img2ImgParams(Params):
    _endpoint_stem = 'img2img'

    pipeline: Literal["stable_diffusion_mega"] = "stable_diffusion_mega"
    pipeline_method: Literal["img2img"] = "img2img"

    initial_image: BlobUrl = pydantic.Field(
        description="The image to use as input for image generation. "
                    "The image must have a width and height divisible by 8. "
    )
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


class InpaintParams(Params):
    _endpoint_stem = 'inpaint'

    pipeline: Literal["stable_diffusion_mega"] = "stable_diffusion_mega"
    pipeline_method: Literal["inpaint"] = "inpaint"

    model: str = pydantic.Field(
        default="runwayml/stable-diffusion-inpainting",
        description="The model to use for image generation, e.g. 'runwayml/stable-diffusion-inpainting'.",
    )

    initial_image: BlobUrl = pydantic.Field(
        description="The image to use as input for image generation. "
                    "It must have a width and height divisible by 8. "
    )
    mask: BlobUrl = pydantic.Field(
        description="The mask to use for image generation. "
                    "It must have the same width and height as the initial image. "
                    "It will be converted to a black-and-white image, "
                    "wherein white indicates the area to be inpainted."
    )


ParamsUnion = Union[tuple(Params.__subclasses__())]  # type: ignore
AnyParams = Union[ParamsUnion, Params]
