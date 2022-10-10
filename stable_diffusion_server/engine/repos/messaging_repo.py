import asyncio
import logging
from collections import defaultdict
from datetime import datetime
from typing import Sequence, Union, AsyncIterable, AsyncIterator

from stable_diffusion_server.engine.utils import get_aioredis, get_redis

logger = logging.getLogger(__name__)


class MessagingRepo:
    ########
    # Pubsub
    ########

    def publish(self, topic: str, message: str) -> None:
        raise NotImplementedError

    async def subscribe(self, topic: str) -> None:
        raise NotImplementedError

    async def listen(self) -> AsyncIterator[str]:
        raise NotImplementedError
        yield  # noqa, without this pyright can't tell that this is a generator, and can't wrap return type

    #######
    # Queue
    #######

    def push(self, queue: str, message: str) -> None:
        raise NotImplementedError

    async def pop(self, queue: Union[str, Sequence[str]]) -> str:
        raise NotImplementedError


_in_memory_topics: dict[str, list[tuple[datetime, str]]] = defaultdict(list)
_in_memory_queues: dict[str, list[str]] = defaultdict(list)


class InMemoryMessagingRepo(MessagingRepo):
    def __init__(self):
        self.subscribed_topics = set()
        self.seen_topic_messages = set()

    def publish(self, topic: str, message: str) -> None:
        _in_memory_topics[topic].append((datetime.now(), message))

        # def timed_remove():
        #     time.sleep(1)
        #     if message in _in_memory_topics[topic]:
        #         _in_memory_topics[topic].remove(message)
        # threading.Thread(target=timed_remove).start()

    async def subscribe(self, topic: str) -> None:
        self.subscribed_topics.add(topic)

    async def listen(self) -> AsyncIterator[str]:
        displayed = defaultdict(list)
        while True:
            for topic in self.subscribed_topics:
                for message in _in_memory_topics[topic]:
                    if message not in displayed[topic]:
                        _, payload = message
                        yield payload
                        displayed[topic].append(message)
            await asyncio.sleep(0.1)

    def push(self, queue: str, message: str) -> None:
        _in_memory_queues[queue].append(message)

    async def pop(self, queue: Union[str, Sequence[str]]) -> str:
        queues = [queue] if isinstance(queue, str) else queue
        while True:
            if not any(_in_memory_queues[q] for q in queues):
                await asyncio.sleep(0.1)
                continue
            for q in queues:
                if _in_memory_queues[q]:
                    return _in_memory_queues[q].pop(0)
            await asyncio.sleep(0.1)


class RedisMessagingRepo(MessagingRepo):
    def __init__(self):
        self.redis = get_redis()
        self.aioredis = get_aioredis()
        self.pubsub = self.aioredis.pubsub()

    def publish(self, topic: str, message: str) -> None:
        num = self.redis.publish(topic, message)
        logger.debug(f'Published message to {topic} for {num} subscribers')

    async def subscribe(self, topic: str) -> None:
        logger.debug(f'Subscribing to topic: {topic}')
        await self.pubsub.subscribe(topic)

    async def listen(self) -> AsyncIterator[str]:
        async for message in self.pubsub.listen():
            if message is None:
                continue
            data = message['data']
            if data == 1:
                continue
            yield data

    def push(self, queue: str, message: str) -> None:
        self.redis.lpush(queue, message)

    async def pop(self, queue: Union[str, Sequence[str]]) -> str:
        _, message = await self.aioredis.brpop(queue)
        return message
