import uuid
from typing import Union, Optional

from typing_extensions import TypeAlias

from stable_diffusion_server.engine.utils import get_redis

BlobId: TypeAlias = str


class BlobRepo:
    def put_blob(self, data: Union[str, bytes]) -> BlobId:
        raise NotImplementedError

    def get_blob(self, blob_id: BlobId) -> Optional[Union[str, bytes]]:
        raise NotImplementedError


class InMemoryBlobRepo(BlobRepo):
    def __init__(self):
        self._blobs = {}

    def put_blob(self, data: Union[str, bytes]) -> BlobId:
        blob_id = BlobId(len(self._blobs))
        self._blobs[blob_id] = data
        return blob_id

    def get_blob(self, blob_id: BlobId) -> Optional[Union[str, bytes]]:
        return self._blobs.get(blob_id, None)


class RedisBlobRepo(BlobRepo):
    def __init__(self):
        self.redis = get_redis()

    def put_blob(self, data: Union[str, bytes]) -> BlobId:
        blob_id = BlobId(uuid.uuid4())
        self.redis.set(blob_id, data)
        return blob_id

    def get_blob(self, blob_id: BlobId) -> Optional[Union[str, bytes]]:
        return self.redis.get(blob_id)
