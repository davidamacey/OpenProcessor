"""The pool a ``cluster_run`` fits: one point-in-time read, keyed and sorted by item key.

Items: every embedded item that is not confidently labelled, validated or shelved (a run
re-clusters the whole pool and identities are carried over by member overlap). Region boxes: the
accepted, vectored boxes (``region_box_rows.scroll_box_rows``), keyed ``crop_id:box_id``.
The vector field is the one the embedding registry resolved for the project, never an env value.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from src.config.curation_accessors import items_index
from src.domain.embedding_state import embedded_clause
from src.domain.ingest_class_sources import confident_class_sources
from src.services.curation.clustering.vector_read import read_vectors


if TYPE_CHECKING:
    from collections.abc import Sequence

    from opensearchpy import AsyncOpenSearch

BOX_KEY_SEPARATOR = ':'
GOOD_BOX_STATES: tuple[str, ...] = ('proposed', 'accepted')


def item_pool_query(vector_field: str, extra_must: Sequence[dict[str, Any]] = ()) -> dict[str, Any]:
    return {
        'bool': {
            'filter': [embedded_clause(vector_field), *extra_must],
            'must_not': [
                {'term': {'class_validated': True}},
                {'terms': {'class_source': list(confident_class_sources())}},
                {'term': {'class_excluded': True}},
            ],
        }
    }


async def snapshot_items(
    client: AsyncOpenSearch, vector_field: str, extra_must: Sequence[dict[str, Any]] = ()
) -> tuple[list[str], np.ndarray]:
    """``(item ids, vectors)`` sorted by id, from one point in time (``read_vectors``)."""
    read = await read_vectors(
        client,
        index=items_index(),
        field=vector_field,
        query=item_pool_query(vector_field, extra_must),
    )
    return read.ids, read.matrix


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
