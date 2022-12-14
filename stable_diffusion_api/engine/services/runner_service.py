import io
import logging
import os
from typing import Any, Optional

import PIL.Image
import numpy as np
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, LMSDiscreteScheduler, StableDiffusionImg2ImgPipeline, \
    StableDiffusionInpaintPipeline, DiffusionPipeline

from stable_diffusion_api.engine.repos.blob_repo import BlobRepo
from stable_diffusion_api.engine.services.event_service import EventService
from stable_diffusion_api.engine.services.status_service import StatusService
from stable_diffusion_api.models.blob import BlobUrl
from stable_diffusion_api.models.events import FinishedEvent, StartedEvent, AbortedEvent
from stable_diffusion_api.models.params import Txt2ImgParams, Img2ImgParams, InpaintParams, Params
from stable_diffusion_api.models.results import GeneratedBlob
from stable_diffusion_api.models.task import Task, TaskId
from stable_diffusion_api.models.user import User, Username

logger = logging.getLogger(__name__)


class TaskCancelledException(Exception):
    pass


class RunnerService:
    def __init__(
        self,
        blob_repo: BlobRepo,
        status_service: StatusService,
        event_service: EventService,
    ):
        self.blob_repo = blob_repo
        self.status_service = status_service
        self.event_service = event_service

        self.cached_kwargs: Optional[dict[str, Any]] = None
        self.cached_pipeline = None

    def get_img(self, blob_url: BlobUrl, is_mask: bool = False):
        # extract image blob into `init_image` pipe kwarg
        blob = self.blob_repo.get_blob(blob_url)
        if blob is None:
            raise ValueError(f'Blob not found: {blob_url}')
        image = PIL.Image.open(io.BytesIO(blob))
        if is_mask:
            # adapted from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint_legacy.preprocess_mask
            # instead of shrinking it down by 8 times, resize it to 64 x 64 (latent space size)
            mask = image.convert('L')
            # w, h = mask.size
            # w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
            mask = mask.resize((64, 64), resample=PIL.Image.NEAREST)
            mask = np.array(mask).astype(np.float32) / 255.0
            mask = np.tile(mask, (4, 1, 1))
            mask = mask[None].transpose(0, 1, 2, 3)  # what does this step do?
            mask = 1 - mask  # repaint white, keep black
            mask = torch.from_numpy(mask)
            return mask
        return image.convert('RGB')

    def pipeline_callback(
        self,
        task_id: TaskId,
        step: int,
        timestep: int,
        latents: torch.FloatTensor,
    ):
        if self.status_service.is_task_cancelled(task_id):
            raise TaskCancelledException()
        # TODO update progress and save intermediate results

    def get_arguments(self, task: Task, device: str) -> tuple[dict[str, Any], dict[str, Any], Optional[str]]:
        params = task.parameters

        # set model
        pipeline_kwargs: dict[str, Any] = {
            'pretrained_model_name_or_path': params.model
        }

        # set token
        if "HUGGINGFACE_TOKEN" in os.environ:
            pipeline_kwargs['use_auth_token'] = os.environ["HUGGINGFACE_TOKEN"]

        # optionally disable safety filter
        if not params.safety_filter:
            pipeline_kwargs['safety_checker'] = None

        # pick scheduler
        match params.scheduler:
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

        # construct generator, set seed if params.seed is not None
        generator = torch.Generator(device)
        if params.seed is None:
            params.seed = generator.seed()
        else:
            generator.manual_seed(params.seed)

        # extract common pipe kwargs
        pipe_kwargs: dict[str, Any] = dict(
            prompt=params.prompt,
            num_inference_steps=params.steps,
            guidance_scale=params.guidance,
            negative_prompt=params.negative_prompt,
            generator=generator,
        )

        # prepare pipeline
        pipeline_kwargs.update(
            custom_pipeline=params._pipeline,
        )
        if isinstance(params, Txt2ImgParams):
            pipe_kwargs.update(
                height=params.height,
                width=params.width,
            )
        elif isinstance(params, Img2ImgParams):
            init_image = self.get_img(params.initial_image)
            pipe_kwargs.update(
                strength=params.strength,
                init_image=init_image,
            )
        elif isinstance(params, InpaintParams):
            init_image = self.get_img(params.initial_image)
            mask_image = self.get_img(params.mask, is_mask=True)
            pipe_kwargs.update(
                init_image=init_image,
                mask_image=mask_image,
            )
        else:
            raise ValueError(f'Unknown params type: {params}')

        return pipeline_kwargs, pipe_kwargs, params._pipeline_method

    def save_img(self, img: PIL.Image.Image, task: Task) -> GeneratedBlob:
        # convert pillow image to png bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()

        # save blob
        blob_url = self.blob_repo.put_blob(img_bytes)

        return GeneratedBlob(
            blob_url=blob_url,
            parameters_used=task.parameters,
        )

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

        # pick device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # extract parameters
        pipeline_kwargs, pipe_kwargs, pipe_method_name = self.get_arguments(task, device)

        try:
            # create or reuse pipeline
            if self.cached_pipeline is not None and self.cached_kwargs == pipeline_kwargs:
                # reuse cached pipeline
                pipe = self.cached_pipeline
            else:
                # create pipeline
                pipe = DiffusionPipeline.from_pretrained(**pipeline_kwargs)
                pipe.to(device)
                self.cached_kwargs = pipeline_kwargs
                self.cached_pipeline = pipe

            # determine pipeline method
            if pipe_method_name is None:
                pipe_method = pipe
            else:
                pipe_method = getattr(pipe, pipe_method_name)

            # run pipeline
            output = pipe_method(
                **pipe_kwargs,
                callback=lambda step, timestep, latents: self.pipeline_callback(task.task_id, step, timestep, latents)
            )

            # get output
            img = output.images[0]
        except TaskCancelledException:
            logger.info(f'Task cancelled by user: {task}')
            self.event_service.send_event(
                task.user.session_id,
                AbortedEvent(
                    event_type="aborted",
                    task_id=task.task_id,
                    reason="Task cancelled by user",
                )
            )
            return
        except Exception as e:
            logger.error(f'Error while handling task: {task}', exc_info=True)
            self.event_service.send_event(
                task.user.session_id,
                AbortedEvent(
                    event_type="aborted",
                    task_id=task.task_id,
                    reason="Internal error: " + str(e),
                )
            )
            return

        # save image
        generated_image = self.save_img(img, task)

        # finished event
        self.event_service.send_event(
            task.user.session_id,
            FinishedEvent(
                event_type="finished",
                task_id=task.task_id,
                result=generated_image,
            )
        )
