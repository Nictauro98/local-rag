from abc import ABC, abstractmethod


class StorageBackend(ABC):
    @abstractmethod
    async def upload(self, filename: str, data: bytes) -> str:
        """Store data under filename; return the filename."""

    @abstractmethod
    async def download(self, filename: str) -> bytes:
        """Return raw bytes for the stored file."""

    @abstractmethod
    async def list(self) -> list[str]:
        """Return all stored filenames."""

    @abstractmethod
    async def exists(self, filename: str) -> bool:
        """Return True if filename is present in storage."""
