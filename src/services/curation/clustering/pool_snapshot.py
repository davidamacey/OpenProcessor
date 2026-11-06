"""The pool a ``cluster_run`` fits: one point-in-time read, keyed and sorted by item key.

Items: every embedded item that is not confidently labelled, validated or shelved (the residual
gate of ``residual_fetch.residual_pool_query``, without its integer ``cluster_id`` clause: a run
re-clusters the whole pool and identities are carried over by member overlap). Region boxes: the
accepted, vectored boxes (``region_box_rows.scroll_box_rows``), keyed ``crop_id:box_id``.
The vector field is the one the embedding registry resolved for the project, never an env value.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from src.config.curation_accessors import items_index
from src.domain.embedding_state import embedded_clause
from src.services.curation.clustering.residual_fetch import CONFIDENT_CLASS_SOURCES


if TYPE_CHECKING:
    from collections.abc import Sequence

    from opensearchpy import AsyncOpenSearch

PAGE = 5000
PIT_KEEP_ALIVE = '5m'
BOX_KEY_SEPARATOR = ':'
GOOD_BOX_STATES: tuple[str, ...] = ('proposed', 'accepted')


def item_pool_query(vector_field: str, extra_must: Sequence[dict[str, Any]] = ()) -> dict[str, Any]:
    return {
        'bool': {
            'filter': [embedded_clause(vector_field), *extra_must],
            'must_not': [
                {'term': {'class_validated': True}},
                {'terms': {'class_source': list(CONFIDENT_CLASS_SOURCES)}},
                {'term': {'class_excluded': True}},
            ],
        }
    }


async def snapshot_items(
    client: AsyncOpenSearch, vector_field: str, extra_must: Sequence[dict[str, Any]] = ()
) -> tuple[list[str], np.ndarray]:
    """``(item ids, vectors)`` sorted by id, from one point in time (PIT + ``search_after``)."""
    index = items_index()
    pit = (await client.create_pit(index=index, keep_alive=PIT_KEEP_ALIVE))['pit_id']
    ids: list[str] = []
    rows: list[list[float]] = []
    after: list[Any] | None = None
    try:
        while True:
            body: dict[str, Any] = {
                'size': PAGE,
                'query': item_pool_query(vector_field, extra_must),
                '_source': [vector_field],
                'pit': {'id': pit, 'keep_alive': PIT_KEEP_ALIVE},
                'sort': [{'_shard_doc': 'asc'}],
            }
            if after is not None:
                body['search_after'] = after
            hits = (await client.search(body=body))['hits']['hits']
            if not hits:
                break
            for hit in hits:
                vector = (hit.get('_source') or {}).get(vector_field)
                if vector:
                    ids.append(hit['_id'])
                    rows.append(vector)
            after = hits[-1]['sort']
    finally:
        await client.delete_pit(body={'pit_id': [pit]})
    if not ids:
        return [], np.zeros((0, 0), dtype=np.float32)
    order = sorted(range(len(ids)), key=ids.__getitem__)
    return [ids[i] for i in order], np.asarray(rows, dtype=np.float32)[order]


async def snapshot_boxes(client: AsyncOpenSearch) -> tuple[list[str], np.ndarray]:
    """``(crop_id:box_id keys, vectors)`` of the good boxes with a current vector, sorted by key."""
    from src.services.curation.clustering.region_box_rows import scroll_box_rows

    rows = await scroll_box_rows(client, index=items_index(), states=GOOD_BOX_STATES)
    keyed = sorted(
        ((f'{r.crop_id}{BOX_KEY_SEPARATOR}{r.box_id}', r.vector) for r in rows), key=lambda p: p[0]
    )
    if not keyed:
        return [], np.zeros((0, 0), dtype=np.float32)
    return [k for k, _ in keyed], np.asarray([v for _, v in keyed], dtype=np.float32)
