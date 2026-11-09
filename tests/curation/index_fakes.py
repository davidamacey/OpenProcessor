"""The ``client.indices`` namespace of an OpenSearch fake: the calls ingest and export make
after their own writes (a refresh, a settings read) answer like a cluster with no such index."""

from __future__ import annotations

from typing import Any


class FakeIndicesApi:
    async def refresh(self, index: str | None = None, **_: Any) -> dict[str, Any]:  # noqa: ARG002
        return {'_shards': {}}

    async def get_settings(self, index: str | None = None, **_: Any) -> dict[str, Any]:  # noqa: ARG002
        return {}

    async def get(self, index: str | None = None, **_: Any) -> dict[str, Any]:  # noqa: ARG002
        return {}

    async def put_mapping(self, index: str | None = None, **_: Any) -> dict[str, Any]:  # noqa: ARG002
        return {'acknowledged': True}

    async def put_settings(self, index: str | None = None, **_: Any) -> dict[str, Any]:  # noqa: ARG002
        return {'acknowledged': True}
