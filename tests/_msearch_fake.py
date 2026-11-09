"""``_msearch`` for the offline fakes: answered from the fake's own ``search`` and ``count``.

Semantic search sends the kNN page and the unembedded count as one ``_msearch``. A fake that
already answers ``search(index=, body=)`` and ``count(index=, body=)`` gets the same behaviour
through ``install_msearch``, so a test keeps asserting on the kNN body and the count scope while
the number of client round trips is what ``fake.msearch.await_count`` says.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock


def install_msearch(fake: Any) -> Any:
    async def msearch(*, body: list[dict[str, Any]]) -> dict[str, Any]:
        responses: list[dict[str, Any]] = []
        for header, request in zip(body[::2], body[1::2], strict=True):
            index = header['index']
            if request.get('size') == 0 and request.get('track_total_hits') is True:
                counted = await fake.count(index=index, body={'query': request['query']})
                total = int((counted or {}).get('count', 0))
                responses.append({'hits': {'total': {'value': total}, 'hits': []}})
            else:
                responses.append(await fake.search(index=index, body=request))
        return {'responses': responses}

    fake.msearch = AsyncMock(side_effect=msearch)
    return fake
