from collections import defaultdict
from typing import Optional

from stable_diffusion_api.engine.utils import get_redis


class KeyValueRepo:
    def store(self, collection: str, key: str, value: str) -> None:
        raise NotImplementedError

    def retrieve(self, collection: str, key: str) -> Optional[str]:
        raise NotImplementedError

    def exists(self, collection: str, key: str) -> bool:
        raise NotImplementedError


class InMemoryKeyValueRepo(KeyValueRepo):
    _store = defaultdict(dict)

    def store(self, collection: str, key: str, value: str) -> None:
        self._store[collection][key] = value

    def retrieve(self, collection: str, key: str) -> Optional[str]:
        return self._store[collection].get(key, None)

    def exists(self, collection: str, key: str) -> bool:
        return key in self._store[collection]


class RedisKeyValueRepo(KeyValueRepo):
    def __init__(self):
        self.redis = get_redis()

    def store(self, collection: str, key: str, value: str) -> None:
        self.redis.hset(collection, key, value)

    def retrieve(self, collection: str, key: str) -> Optional[str]:
        v = self.redis.hget(collection, key)
        if v is None:
            return None
        return v.decode('utf-8')

    def exists(self, collection: str, key: str) -> bool:
        return self.redis.hexists(collection, key)
