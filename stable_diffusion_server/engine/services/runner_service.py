import io
import logging
import os
from typing import Mapping, Any

import PIL.Image
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, LMSDiscreteScheduler, StableDiffusionImg2ImgPipeline

from stable_diffusion_server.engine.repos.blob_repo import BlobRepo
from stable_diffusion_server.engine.services.event_service import EventService
from stable_diffusion_server.models.events import FinishedEvent, StartedEvent, CancelledEvent
from stable_diffusion_server.models.image import GeneratedImage
from stable_diffusion_server.models.params import Txt2ImgParams, Img2ImgParams
from stable_diffusion_server.models.task import Task

logger = logging.getLogger(__name__)


class RunnerService:
    def __init__(
        self,
        blob_repo: BlobRepo,
        event_service: EventService
    ):
        self.blob_repo = blob_repo
        self.event_service = event_service

    async def run_task(self, task: Task) -> None:
        logger.info(f'Handle task: {task}')

        # started event
        self.event_service.send_event(
            task.user.session_id,
            StartedEvent(
                event_type="started",
                task_id=task.task_id,
            )
        )

        pipeline_kwargs: dict[str, Any] = {}

        # set token
        if "HUGGINGFACE_TOKEN" in os.environ:
            pipeline_kwargs['use_auth_token'] = os.environ["HUGGINGFACE_TOKEN"]

        # optionally disable safety filter
        if not task.parameters.safety_filter:
            pipeline_kwargs['safety_checker'] = None

        # pick scheduler
        match task.parameters.scheduler:
            case "plms":
                pass  # default scheduler
            case "ddim":
                pipeline_kwargs['scheduler'] = DDIMScheduler(
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                    clip_sample=False,
                    set_alpha_to_one=False
                )
            case "k-lms":
                pipeline_kwargs['scheduler'] = LMSDiscreteScheduler(
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear"
                )

        # handle task
        params = task.parameters
        try:
            if isinstance(params, Txt2ImgParams):
                img = self._handle_txt2img_task(params, pipeline_kwargs)
            elif isinstance(params, Img2ImgParams):
                img = self._handle_img2img_task(params, pipeline_kwargs)
            else:
                raise NotImplementedError(f'Unknown task type: {params.task_type}')
        except Exception as e:
            logger.error(f'Error while handling task: {task}', exc_info=True)
            self.event_service.send_event(
                task.user.session_id,
                CancelledEvent(
                    event_type="cancelled",
                    task_id=task.task_id,
                    reason="Internal error: " + str(e),
                )
            )
            return

        # convert pillow image to png bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()

        blob_id = self.blob_repo.put_blob(img_bytes)

        generated_image = GeneratedImage(
            blob_id=blob_id,
            parameters_used=params,
        )

        # finished event
        self.event_service.send_event(
            task.user.session_id,
            FinishedEvent(
                event_type="finished",
                task_id=task.task_id,
                image=generated_image,
            )
        )

    def _handle_txt2img_task(self, params: Txt2ImgParams, pipeline_kwargs: Mapping[str, Any]) -> PIL.Image.Image:
        pipe = StableDiffusionPipeline.from_pretrained(
            params.model_id,
            **pipeline_kwargs
        )

        pipe = pipe.to("cpu")  # TODO: use GPU if available

        output = pipe(
            prompt=params.prompt,
            height=params.height,
            width=params.width,
            num_inference_steps=params.steps,
            guidance_scale=params.guidance,
            negative_prompt=params.negative_prompt,
        )
        return output.images[0]

    def _handle_img2img_task(self, params: Img2ImgParams, pipeline_kwargs: Mapping[str, Any]) -> PIL.Image.Image:
        # pull image
        blob = self.blob_repo.get_blob(params.initial_image)
        if blob is None:
            raise RuntimeError(f'Blob not found: {params.initial_image}')
        image = PIL.Image.open(io.BytesIO(blob))

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            params.model_id,
            **pipeline_kwargs,
        )

        pipe = pipe.to("cpu")  # TODO: use GPU if available

        output = pipe(
            prompt=params.prompt,
            init_image=image,
            strength=params.strength,
            num_inference_steps=params.steps,
            guidance_scale=params.guidance,
            negative_prompt=params.negative_prompt,
        )
        return output.images[0]
