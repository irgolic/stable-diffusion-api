import logging

from stable_diffusion_server.engine.repos.blob_repo import BlobRepo
from stable_diffusion_server.engine.services.event_service import EventService
from stable_diffusion_server.models.events import FinishedEvent, StartedEvent
from stable_diffusion_server.models.image import GeneratedImage
from stable_diffusion_server.models.task import Task, Txt2ImgTask, Img2ImgTask

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

        # handle task
        if isinstance(task, Txt2ImgTask):
            generated_img = self._handle_txt2img_task(task)
        elif isinstance(task, Img2ImgTask):
            generated_img = self._handle_img2img_task(task)
        else:
            raise NotImplementedError(f'Unknown task type: {task}')

        # finished event
        self.event_service.send_event(
            task.user.session_id,
            FinishedEvent(
                event_type="finished",
                task_id=task.task_id,
                image=generated_img,
            )
        )

    def _handle_txt2img_task(self, task: Txt2ImgTask) -> GeneratedImage:
        ...  # TODO do some processing

        return GeneratedImage(
            link="dummy",
            format="dummy",
            parameters_used=task.params,
        )

    def _handle_img2img_task(self, task: Img2ImgTask) -> GeneratedImage:
        ...  # TODO do some processing

        return GeneratedImage(
            link="dummy",
            format="dummy",
            parameters_used=task.params,
        )
