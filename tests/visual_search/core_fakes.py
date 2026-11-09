"""An OpenSearch stand-in for core ingest: the global index in memory, the writes of the ingest
paths and the calls the grouping job and the candidate lookup make."""

from __future__ import annotations

from typing import Any

import numpy as np
from near_duplicates.fake_opensearch import FakeNearDupOpenSearch


INDEX = 'visual_search_global'


class FakeCoreOpenSearch(FakeNearDupOpenSearch):
    """The calls the core ingest, the candidate lookup and the grouping job make, over one
    in-memory global index."""

    def __init__(self) -> None:
        super().__init__({INDEX: {}})
        self.client = self

    async def check_duplicates_by_hash_batch(self, hashes: list[str]) -> dict[str, Any]:
        return dict.fromkeys(hashes)

    async def check_duplicate_by_hash(self, _imohash: str) -> None:
        return None

    async def bulk_ingest(self, documents: list[dict[str, Any]], refresh: bool = False) -> Any:  # noqa: ARG002
        for doc in documents:
            self.docs[INDEX][doc['image_id']] = {
                **doc,
                'global_embedding': np.asarray(doc['global_embedding']).tolist(),
            }
        return {'global': len(documents)}

    async def bulk_index_faces(self, _faces: list[dict[str, Any]]) -> Any:
        return {'indexed': 0}

    async def ingest_image(self, **kw: Any) -> dict[str, Any]:
        self.docs[INDEX][kw['image_id']] = {
            'image_id': kw['image_id'],
            'global_embedding': np.asarray(kw['global_embedding']).tolist(),
        }
        return {'global': True, 'vehicles': 0, 'people': 0, 'skipped': 0, 'errors': []}
