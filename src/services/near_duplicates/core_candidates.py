"""The near-duplicate candidates a core ingest request reports: stored images above the core cut.

A request never groups (``near_dup_core_group`` does, once the pending rows are claimed); it only
reads. One ``_msearch`` over the core global index answers every image of the batch: the ``k``
nearest neighbours of each, the exact cosine computed from the stored vectors, the images at or
above ``DEFAULT_CORE_DEDUP_COSINE`` kept. Images of the same request are not in the index yet
(the bulk write does not refresh), so a candidate is always an image stored before it.

Docs: OpenSearch k-NN query and multi-search
(https://docs.opensearch.org/docs/latest/vector-search/vector-search-techniques/knn-core-concepts/).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.clients.opensearch.names import IndexName
from src.domain.near_dup import DEFAULT_CORE_DEDUP_COSINE
from src.services.near_duplicates import candidates as knn


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from opensearchpy import AsyncOpenSearch

CANDIDATE_K = 10
"""Neighbours read per image (the request lists at most this many candidates)."""


@dataclass(frozen=True)
class Candidate:
    image_id: str
    cosine: float


async def find_candidates(
    client: AsyncOpenSearch, vectors: Mapping[str, Sequence[float] | np.ndarray]
) -> dict[str, list[Candidate]]:
    """``image id -> candidates`` (closest first, id order on a tie) for each image that has any.

    An image with no embedding (CLIP disabled for the request) is skipped.
    """
    members = {
        image_id: knn.unit_rows(np.asarray([vector], dtype=np.float32))[0]
        for image_id, vector in vectors.items()
        if len(vector) and np.any(vector)
    }
    edges = await knn.knn_edges(
        client,
        index=IndexName.GLOBAL.value,
        field='global_embedding',
        members=members,
        k=CANDIDATE_K,
        min_cosine=DEFAULT_CORE_DEDUP_COSINE,
    )
    found: dict[str, list[Candidate]] = {}
    for a, b, cosine in edges:
        for mine, other in ((a, b), (b, a)):
            if mine in members:
                found.setdefault(mine, []).append(Candidate(other, cosine))
    return {
        image_id: sorted(items, key=lambda c: (-c.cosine, c.image_id))
        for image_id, items in found.items()
    }
