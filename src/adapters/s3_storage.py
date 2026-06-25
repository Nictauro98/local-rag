"""
AWS S3 storage adapter — migration stub.

Migration steps:
1. Add boto3 to dependencies: `uv add boto3`.
2. Set env vars: RAG_AWS_REGION, RAG_AWS_ACCESS_KEY_ID, RAG_AWS_SECRET_ACCESS_KEY (or use IAM role).
3. Replace self._client = boto3.client("s3", region_name=...) and wrap all SDK calls in asyncio.to_thread.
4. upload   → s3.put_object(Bucket=bucket, Key=filename, Body=data)
5. download → s3.get_object(Bucket=bucket, Key=filename)["Body"].read()
6. list     → s3.list_objects_v2(Bucket=bucket) paginator
7. exists   → s3.head_object(Bucket=bucket, Key=filename); catch ClientError 404
8. Switch RAG_STORAGE_BACKEND=aws in .env.
"""

from src.interfaces.storage import StorageBackend


class S3Storage(StorageBackend):
    async def upload(self, filename: str, data: bytes) -> str:
        raise NotImplementedError("S3Storage is a migration stub. See module docstring.")

    async def download(self, filename: str) -> bytes:
        raise NotImplementedError("S3Storage is a migration stub. See module docstring.")

    async def list(self) -> list[str]:
        raise NotImplementedError("S3Storage is a migration stub. See module docstring.")

    async def exists(self, filename: str) -> bool:
        raise NotImplementedError("S3Storage is a migration stub. See module docstring.")
