import os
import typing
import uuid
from typing import Union, Optional

from jose import jwt
import jose.exceptions
from typing_extensions import TypeAlias

from stable_diffusion_api.engine.utils import get_redis
from stable_diffusion_api.models.blob import BlobUrl, BlobId, BlobToken
from stable_diffusion_api.models.user import Username
import requests


class BlobRepo:
    def put_blob(self, blob: bytes) -> BlobUrl:
        raise NotImplementedError

    def get_blob(self, blob_url: BlobUrl) -> Optional[bytes]:
        return requests.get(blob_url).content


class LocalBlobRepo(BlobRepo):
    def __init__(
        self,
        base_blob_url: str,
        secret_key: str,
        algorithm: str,
    ):
        self.base_blob_url = base_blob_url
        self.secret_key = secret_key
        self.algorithm = algorithm

    def __make_token(self, blob_id: BlobId) -> BlobToken:
        return BlobToken(jwt.encode({"blob_id": blob_id},
                                    self.secret_key,
                                    algorithm=self.algorithm))

    def __verify_token(self, blob_token: BlobToken) -> Optional[BlobId]:
        try:
            return BlobId(jwt.decode(blob_token,
                                     self.secret_key,
                                     algorithms=[self.algorithm])["blob_id"])
        except jose.exceptions.JWTError:
            return None

    def __make_url(self, blob_id: BlobId) -> BlobUrl:
        blob_token = self.__make_token(blob_id)

        return BlobUrl(f"{self.base_blob_url}/{blob_token}")

    def put_blob(self, blob: bytes) -> BlobUrl:
        blob_id = BlobId(uuid.uuid4())
        self._store_blob(blob_id, blob)
        return self.__make_url(blob_id)

    def get_blob_by_token(self, blob_token: BlobToken) -> Optional[bytes]:
        blob_id = self.__verify_token(blob_token)
        if blob_id is None:
            return None
        return self._retrieve_blob(blob_id)

    def get_blob(self, blob_url: BlobUrl) -> Optional[bytes]:
        if not blob_url.startswith(self.base_blob_url):
            return super().get_blob(blob_url)
        blob_token = blob_url.removeprefix(self.base_blob_url + "/")

        return self.get_blob_by_token(blob_token)

    def _store_blob(self, blob_id: BlobId, blob: bytes) -> None:
        raise NotImplementedError

    def _retrieve_blob(self, blob_id: BlobId) -> Optional[bytes]:
        raise NotImplementedError


class InMemoryBlobRepo(LocalBlobRepo):
    _blobs = {}

    def _store_blob(self, blob_id: BlobId, blob: bytes) -> None:
        self._blobs[blob_id] = blob

    def _retrieve_blob(self, blob_id: BlobId) -> Optional[bytes]:
        return self._blobs.get(blob_id, None)


class RedisBlobRepo(LocalBlobRepo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.redis = get_redis()

    def _store_blob(self, blob_id: BlobId, blob: bytes) -> None:
        self.redis.hset('blob_data', blob_id, blob)

    def _retrieve_blob(self, blob_id: BlobId) -> Optional[bytes]:
        data = self.redis.hget('blob_data', blob_id)
        if data is None:
            return None
        return data
