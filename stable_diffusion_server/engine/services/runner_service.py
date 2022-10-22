import io
import logging
import os
from typing import Any, Optional

import PIL.Image
import numpy as np
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, LMSDiscreteScheduler, StableDiffusionImg2ImgPipeline, \
    StableDiffusionInpaintPipeline

from stable_diffusion_server.engine.repos.blob_repo import BlobRepo
from stable_diffusion_server.engine.services.event_service import EventService
from stable_diffusion_server.models.blob import Blob, BlobId
from stable_diffusion_server.models.events import FinishedEvent, StartedEvent, CancelledEvent
from stable_diffusion_server.models.image import GeneratedImage
from stable_diffusion_server.models.params import Txt2ImgParams, Img2ImgParams, InpaintParams
from stable_diffusion_server.models.task import Task
from stable_diffusion_server.models.user import User, Username

logger = logging.getLogger(__name__)


class RunnerService:
    def __init__(
        self,
        blob_repo: BlobRepo,
        event_service: EventService
    ):
        self.blob_repo = blob_repo
        self.event_service = event_service

    def get_img(self, blob_id: BlobId, username: Username, is_mask: bool = False):
        # extract image blob into `init_image` pipe kwarg
        blob = self.blob_repo.get_blob(blob_id, username)
        if blob is None:
            raise RuntimeError(f'Blob not found: {blob_id}')
        image = PIL.Image.open(io.BytesIO(blob.data))
        image = image.convert('RGB')  # remove alpha channel
        if is_mask:
            # adapted from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint_legacy.preprocess_mask
            # instead of shrinking it down by 8 times, resize it to 64 x 64 (latent space size)
            mask = image.convert('L')
            w, h = mask.size
            w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
            mask = mask.resize((64, 64), resample=PIL.Image.NEAREST)
            mask = np.array(mask).astype(np.float32) / 255.0
            mask = np.tile(mask, (4, 1, 1))
            mask = mask[None].transpose(0, 1, 2, 3)  # what does this step do?
            mask = 1 - mask  # repaint white, keep black
            mask = torch.from_numpy(mask)
            return mask
        else:
            image = image.convert('RGB')
        return image

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

        params = task.parameters

        # set model
        pipeline_kwargs: dict[str, Any] = {
            'pretrained_model_name_or_path': params.model_id
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

        # pick device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # construct generator, set seed if params.seed is not None
        generator = torch.Generator(device)
        if params.seed is None:
            params.seed = generator.seed()
        else:
            generator.manual_seed(params.seed)

        # extract common pipe kwargs
        pipe_kwargs = dict(
            pretrained_model_name_or_path=params.model_id,
            prompt=params.prompt,
            num_inference_steps=params.steps,
            guidance_scale=params.guidance,
            negative_prompt=params.negative_prompt,
        )

        # prepare pipeline
        if isinstance(params, Txt2ImgParams):
            pipeline = StableDiffusionPipeline
            pipe_kwargs.update(
                height=params.height,
                width=params.width,
            )
        elif isinstance(params, Img2ImgParams):
            pipeline = StableDiffusionImg2ImgPipeline
            pipe_kwargs.update(
                strength=params.strength,
                init_image=self.get_img(params.initial_image, task.user.username),
            )
        elif isinstance(params, InpaintParams):
            pipeline = StableDiffusionInpaintPipeline
            pipe_kwargs.update(
                init_image=self.get_img(params.initial_image, task.user.username),
                mask_image=self.get_img(params.mask, task.user.username, is_mask=True),
            )
        else:
            raise NotImplementedError(f'Unknown task type: {params.task_type}')

        # run pipeline
        try:
            pipe = pipeline.from_pretrained(**pipeline_kwargs)
            pipe.to(device)
            output = pipe(**pipe_kwargs)
            img = output.images[0]
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

        # save blob
        blob = Blob(
            data=img_bytes,
            username=task.user.username,
        )
        blob_id = self.blob_repo.put_blob(blob)

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
