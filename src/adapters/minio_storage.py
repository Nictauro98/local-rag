import asyncio
import io

from minio import Minio
from minio.error import S3Error

from src.interfaces.storage import StorageBackend


class MinIOStorage(StorageBackend):
    def __init__(
        self, endpoint: str, access_key: str, secret_key: str, bucket: str, secure: bool = False
    ):
        self._client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        self._bucket = bucket
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        if not self._client.bucket_exists(self._bucket):
            self._client.make_bucket(self._bucket)

    async def upload(self, filename: str, data: bytes) -> str:
        await asyncio.to_thread(
            self._client.put_object,
            self._bucket,
            filename,
            io.BytesIO(data),
            len(data),
        )
        return filename

    async def download(self, filename: str) -> bytes:
        response = await asyncio.to_thread(self._client.get_object, self._bucket, filename)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    async def list(self) -> list[str]:
        objects = await asyncio.to_thread(self._client.list_objects, self._bucket)
        return [obj.object_name for obj in objects]

    async def exists(self, filename: str) -> bool:
        try:
            await asyncio.to_thread(self._client.stat_object, self._bucket, filename)
            return True
        except S3Error:
            return False
