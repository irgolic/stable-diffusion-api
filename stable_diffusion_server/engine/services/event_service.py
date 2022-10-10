import logging
from typing import AsyncIterator

import pydantic

from stable_diffusion_server.engine.repos.messaging_repo import MessagingRepo
from stable_diffusion_server.engine.utils import _serialize_message, _deserialize_message
from stable_diffusion_server.models.events import EventUnion
from stable_diffusion_server.models.user import SessionId

logger = logging.getLogger(__name__)


class EventService:
    def __init__(
        self,
        messaging_repo: MessagingRepo,
    ):
        self.messaging_repo = messaging_repo

    def send_event(self, session_id: SessionId, event: EventUnion):
        msg = _serialize_message(session_id, event)
        self.messaging_repo.publish('event', msg)


class EventListener:
    def __init__(
        self,
        messaging_repo: MessagingRepo,
    ):
        self.messaging_repo = messaging_repo

    async def listen(self) -> AsyncIterator[tuple[SessionId, EventUnion]]:
        await self.messaging_repo.subscribe('event')
        async for data in self.messaging_repo.listen():
            if data is None:
                continue
            yield _deserialize_message(data, EventUnion)
