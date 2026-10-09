"""Batched ``_mget`` for the items index."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.clients.curation_opensearch.base import config


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch


async def mget_crops(
    client: AsyncOpenSearch,
    crop_ids: list[str],
    *,
    index: str | None = None,
    source_includes: list[str] | None = None,
    source_excludes: list[str] | None = None,
    seq_no: bool = False,  # noqa: ARG001 - documents caller intent; mget always returns seq_no/primary_term
    chunk_size: int = 256,
) -> dict[str, dict[str, Any]]:
    """Batched ``_mget`` for the items index.

    Replaces per-item ``await client.get(...)`` loops with a single
    round-trip per chunk of up to ``chunk_size`` ids.

    Args:
        client: AsyncOpenSearch instance.
        crop_ids: list of item ids to fetch. Missing ids are silently
            omitted from the result (no KeyError).
        index: target index. Defaults to the bound project's items
            index (resolved per call); pass explicitly to mget against a
            different index.
        source_includes: if set, restricts the ``_source`` returned
            (keeps the response small). ``None`` returns the full doc.
        source_excludes: if set, drops these fields from ``_source``
            (e.g. large embedding vectors) while keeping everything
            else. Combined with ``source_includes`` only if both are
            given (OpenSearch honors both on the same ``_source``
            clause).
        seq_no: if True, ensures ``_seq_no``/``_primary_term`` come back
            on each doc for OCC. ``mget`` always returns
            ``_seq_no``/``_primary_term`` at the top level of each
            per-doc response regardless of the ``_source`` filter, so
            this flag exists purely for callers to document intent; no
            special body is required, but we keep the param for API
            stability.
        chunk_size: max ids per ``_mget`` call. OS limits the request
            body size; 256 is a safe ceiling.

    Returns:
        ``{crop_id: doc}`` where ``doc`` is the OpenSearch response
        (``_source`` + ``_seq_no``/``_primary_term``).
    """
    if not crop_ids:
        return {}
    if index is None:
        index = config.items_index

    source_clause: Any = None
    if source_includes is not None or source_excludes is not None:
        source_clause = {}
        if source_includes is not None:
            source_clause['includes'] = source_includes
        if source_excludes is not None:
            source_clause['excludes'] = source_excludes

    out: dict[str, dict[str, Any]] = {}
    for start in range(0, len(crop_ids), chunk_size):
        chunk = crop_ids[start : start + chunk_size]
        body: dict[str, Any] = {
            'docs': [{'_id': cid, '_index': index} for cid in chunk],
        }
        if source_clause is not None:
            for doc in body['docs']:
                doc['_source'] = source_clause
        resp = await client.mget(body=body)
        for d in resp.get('docs', []):
            if not d.get('found'):
                continue
            out[d['_id']] = d
    return out
