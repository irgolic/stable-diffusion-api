import uuid
from typing import Union, Optional

from typing_extensions import TypeAlias

from stable_diffusion_server.engine.utils import get_redis

BlobId: TypeAlias = str


class BlobRepo:
    def put_blob(self, data: bytes) -> BlobId:
        raise NotImplementedError

    def get_blob(self, blob_id: BlobId) -> Optional[bytes]:
        raise NotImplementedError

    def get_blob_url(self, blob_id: BlobId) -> str:
        raise NotImplementedError


class InMemoryBlobRepo(BlobRepo):
    def __init__(self):
        self._blobs = {}

    def put_blob(self, data: bytes) -> BlobId:
        blob_id = BlobId(len(self._blobs))
        self._blobs[blob_id] = data
        return blob_id

    def get_blob(self, blob_id: BlobId) -> Optional[bytes]:
        return self._blobs.get(blob_id, None)

    def get_blob_url(self, blob_id: BlobId) -> str:
        # TODO serve from memory
        return f'/blob/{blob_id}'


class RedisBlobRepo(BlobRepo):
    def __init__(self):
        self.redis = get_redis()

    def put_blob(self, data: bytes) -> BlobId:
        blob_id = BlobId(uuid.uuid4())
        self.redis.set(blob_id, data)
        return blob_id

    def get_blob(self, blob_id: BlobId) -> Optional[bytes]:
        return self.redis.get(blob_id)

    def get_blob_url(self, blob_id: BlobId) -> str:
        # TODO serve from memory
        return f'/blob/{blob_id}'
