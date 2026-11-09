"""An in-memory stand-in for the redis-py asyncio client calls the API makes outside the event feed."""

from __future__ import annotations


class FakeValkey:
    """``set``, ``get`` and ``aclose`` over a dict that several clients can share."""

    def __init__(self, data: dict[str, tuple[str, int | None]] | None = None) -> None:
        self.data: dict[str, tuple[str, int | None]] = {} if data is None else data

    async def set(self, key: str, value: str, ex: int | None = None) -> bool:
        self.data[key] = (value, ex)
        return True

    async def get(self, key: str) -> str | None:
        found = self.data.get(key)
        return None if found is None else found[0]

    async def aclose(self) -> None:
        return None
