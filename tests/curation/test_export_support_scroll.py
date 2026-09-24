"""F-26 coverage for ``src.services.curation.export_support.scroll_hits``.

Default page size bump (500 -> 2000): export ``_source`` is ~8 small
fields, so a bigger scroll page is safe and cuts round trips on large
exports. Does NOT switch to docvalue_fields (would corrupt multi-valued
numeric bbox_norm arrays, which come back sorted+deduplicated).
"""

from __future__ import annotations

from typing import Any

import pytest

from src.services.curation.export_support import scroll_hits


class _FakeScrollOS:
    def __init__(self) -> None:
        self.search_bodies: list[dict[str, Any]] = []

    async def search(self, *, index: str, body: dict[str, Any], **kw: Any) -> dict[str, Any]:  # noqa: ARG002
        self.search_bodies.append(body)
        return {'_scroll_id': None, 'hits': {'hits': []}}


@pytest.mark.asyncio
async def test_scroll_hits_default_page_size_is_2000() -> None:
    client = _FakeScrollOS()
    await scroll_hits(client, index='op_items', query={'match_all': {}}, source=['crop_id'])
    assert client.search_bodies[0]['size'] == 2000
