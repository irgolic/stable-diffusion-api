import os
import typing

import aioredis
import pydantic
import redis

from stable_diffusion_api.models.user import SessionId

T = typing.TypeVar('T', bound=pydantic.BaseModel)


# redis


def load_redis():
    host = os.environ.get('REDIS_HOST', 'localhost')
    port = int(os.environ.get('REDIS_PORT', 6379))
    password = os.environ.get('REDIS_PASSWORD', None)
    r = redis.Redis(
        host=host,
        port=port,
        password=password,
    )
    r.ping()
    return r


redis_client = None


def get_redis():
    global redis_client
    if redis_client is None:
        redis_client = load_redis()
    return redis_client


aioredis_client = None


def load_aioredis():
    host = os.environ.get('REDIS_HOST', 'localhost')
    port = int(os.environ.get('REDIS_PORT', 6379))
    password = os.environ.get('REDIS_PASSWORD', None)
    return aioredis.Redis(
        host=host,
        port=port,
        password=password,
    )


def get_aioredis():
    global aioredis_client
    if aioredis_client is None:
        aioredis_client = load_aioredis()
    return aioredis_client


# serialize/deserialize session id with events


def _serialize_message(session_id: SessionId, model: pydantic.BaseModel) -> str:
    return "{" + f'"{session_id}": {model.json()}' + "}"


def _deserialize_message(message: str, model_class: typing.Type[T]) -> tuple[SessionId, T]:
    session_id, event = next(iter(pydantic.parse_raw_as(dict[SessionId, model_class], message).items()))
    return session_id, event
