import uuid
from typing import Union, Optional

from typing_extensions import TypeAlias

from stable_diffusion_server.engine.utils import get_redis
from stable_diffusion_server.models.blob import Blob, BlobId
from stable_diffusion_server.models.user import Username


class BlobRepo:
    def put_blob(self, blob: Blob) -> BlobId:
        blob_id = BlobId(uuid.uuid4())
        self._store_blob(blob_id, blob)
        return blob_id

    def get_blob(self, blob_id: BlobId, username: Username) -> Optional[Blob]:
        blob = self._retrieve_blob(blob_id)
        if blob is None:
            return None
        if blob.username != username:
            return None
        return blob

    def _store_blob(self, blob_id: BlobId, blob: Blob) -> None:
        raise NotImplementedError

    def _retrieve_blob(self, blob_id: BlobId) -> Optional[Blob]:
        raise NotImplementedError


class InMemoryBlobRepo(BlobRepo):
    _blobs = {}

    def _store_blob(self, blob_id: BlobId, blob: Blob) -> None:
        self._blobs[blob_id] = blob

    def _retrieve_blob(self, blob_id: BlobId) -> Optional[Blob]:
        return self._blobs.get(blob_id, None)


class RedisBlobRepo(BlobRepo):
    def __init__(self):
        self.redis = get_redis()

    def _store_blob(self, blob_id: BlobId, blob: Blob) -> None:
        self.redis.hset('blob_data', blob_id, blob.data)
        self.redis.hset('blob_username', blob_id, blob.username.encode('utf-8'))

    def _retrieve_blob(self, blob_id: BlobId) -> Optional[Blob]:
        data = self.redis.hget('blob_data', blob_id)
        username = self.redis.hget('blob_username', blob_id)
        if data is None or username is None:
            return None
        return Blob(
            data=data,
            username=username.decode('utf-8'),
        )
