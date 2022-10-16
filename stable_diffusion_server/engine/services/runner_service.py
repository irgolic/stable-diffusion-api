import io
import logging
import os
from typing import Mapping, Any

import PIL.Image
from diffusers import StableDiffusionPipeline, DDIMScheduler, LMSDiscreteScheduler, StableDiffusionImg2ImgPipeline

from stable_diffusion_server.engine.repos.blob_repo import BlobRepo
from stable_diffusion_server.engine.services.event_service import EventService
from stable_diffusion_server.models.events import FinishedEvent, StartedEvent
from stable_diffusion_server.models.image import GeneratedImage, Image
from stable_diffusion_server.models.task import Task, Txt2ImgTask, Img2ImgTask, TaskUnion

logger = logging.getLogger(__name__)


class RunnerService:
    def __init__(
        self,
        blob_repo: BlobRepo,
        event_service: EventService
    ):
        self.blob_repo = blob_repo
        self.event_service = event_service

    async def run_task(self, task: TaskUnion) -> None:
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
        try:
            if isinstance(task, Txt2ImgTask):
                img = self._handle_txt2img_task(task, pipeline_kwargs)
            elif isinstance(task, Img2ImgTask):
                img = self._handle_img2img_task(task, pipeline_kwargs)
            else:
                raise NotImplementedError(f'Unknown task type: {task}')
        except Exception as e:
            print(e)
            return

        # convert pillow image to png bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()

        blob_id = self.blob_repo.put_blob(img_bytes)
        blob_url = self.blob_repo.get_blob_url(blob_id)

        generated_image = GeneratedImage(
            blob_id=blob_id,
            format="png",
            parameters_used=task.parameters,
            link=blob_url,
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

    def _handle_txt2img_task(self, task: Txt2ImgTask, pipeline_kwargs: Mapping[str, Any]) -> PIL.Image.Image:
        pipe = StableDiffusionPipeline.from_pretrained(
            task.parameters.model_id,
            **pipeline_kwargs
        )

        pipe = pipe.to("cpu")  # TODO: use GPU if available

        output = pipe(
            prompt=task.parameters.prompt,
            height=task.parameters.height,
            width=task.parameters.width,
            num_inference_steps=task.parameters.steps,
            guidance_scale=task.parameters.guidance,
            negative_prompt=task.parameters.negative_prompt,
        )
        return output.images[0]

    def _handle_img2img_task(self, task: Img2ImgTask, pipeline_kwargs: Mapping[str, Any]) -> PIL.Image.Image:
        # pull image
        blob = self.blob_repo.get_blob(task.input_image.blob_id)

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            task.parameters.model_id,
            **pipeline_kwargs,
        )

        pipe = pipe.to("cpu")  # TODO: use GPU if available

        output = pipe(
            prompt=task.parameters.prompt,
            image=blob,
            strength=task.parameters.strength,
            num_inference_steps=task.parameters.steps,
            guidance_scale=task.parameters.guidance,
            negative_prompt=task.parameters.negative_prompt,
        )
        return output.images[0]
