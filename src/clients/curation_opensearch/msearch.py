"""One ``_msearch`` over a single index with every per-entry error raised.

OpenSearch answers a multi-search with 200 even when an entry failed, so each response is checked.
https://docs.opensearch.org/latest/api-reference/multi-search/
"""

from __future__ import annotations

from typing import Any


async def msearch_checked(
    opensearch: Any, index: str, bodies: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """The responses of ``bodies`` searched against ``index``, in order; raises ``RuntimeError``
    when the answer is short or any entry carries an ``error``."""
    payload: list[dict[str, Any]] = []
    for body in bodies:
        payload.extend(({'index': index}, body))
    resp = await opensearch.msearch(body=payload)
    responses: list[dict[str, Any]] = resp.get('responses') or []
    if len(responses) != len(bodies):
        raise RuntimeError(f'_msearch answered {len(responses)} of {len(bodies)} searches')
    for entry in responses:
        if entry.get('error'):
            raise RuntimeError(f'_msearch entry failed: {entry["error"]}')
    return responses
