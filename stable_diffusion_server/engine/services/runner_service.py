import logging

from stable_diffusion_server.engine.repos.blob_repo import BlobRepo
from stable_diffusion_server.engine.services.event_service import EventService
from stable_diffusion_server.models.events import FinishedEvent
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

        if isinstance(task, Txt2ImgTask):
            self._handle_txt2img_task(task)
        elif isinstance(task, Img2ImgTask):
            self._handle_img2img_task(task)
        else:
            raise NotImplementedError(f'Unknown task type: {task}')

    def _handle_txt2img_task(self, task: Txt2ImgTask) -> None:
        ...  # TODO do some processing

        self.event_service.send_event(
            task.session_id,
            FinishedEvent(
                event_type="finished",
                task_id=task.task_id,
                image=GeneratedImage(
                    link="dummy",
                    format="dummy",
                    parameters_used=task.params,
                ),
            )
        )

    def _handle_img2img_task(self, task: Img2ImgTask) -> None:
        ...  # TODO do some processing

        self.event_service.send_event(
            task.session_id,
            FinishedEvent(
                event_type="finished",
                task_id=task.task_id,
                image=GeneratedImage(
                    link="dummy",
                    format="dummy",
                    parameters_used=task.params,
                ),
            )
        )
