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


class FakeValkeySets:
    """The set commands the image-count dirty set uses, over a dict shared by several clients."""

    def __init__(self, data: dict[str, set[str]] | None = None) -> None:
        self.data: dict[str, set[str]] = {} if data is None else data

    async def sadd(self, key: str, *members: str) -> int:
        members_set = self.data.setdefault(key, set())
        before = len(members_set)
        members_set.update(members)
        return len(members_set) - before

    async def exists(self, key: str) -> int:
        return int(bool(self.data.get(key)))

    async def rename(self, source: str, destination: str) -> bool:
        import redis

        if source not in self.data:
            raise redis.ResponseError('ERR no such key')
        self.data[destination] = self.data.pop(source)
        return True

    async def srandmember(self, key: str, count: int) -> list[str]:
        return sorted(self.data.get(key, set()))[:count]

    async def srem(self, key: str, *members: str) -> int:
        current = self.data.get(key, set())
        removed = current & set(members)
        current -= removed
        if not current:
            self.data.pop(key, None)
        return len(removed)

    async def aclose(self) -> None:
        return None
