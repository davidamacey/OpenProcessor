"""The per-index create bodies carry the shared knn settings and the per-index HNSW vector field."""

from __future__ import annotations

from typing import Any

import pytest

from src.clients.opensearch.indexes import IndexesMixin


class _Indices:
    async def put_mapping(self, **_kwargs: Any) -> dict[str, Any]:
        return {}


class _Client:
    indices = _Indices()


class _Recorder(IndexesMixin):
    embedding_dim = 512
    client = _Client()  # type: ignore[assignment]

    def __init__(self) -> None:
        self.bodies: dict[str, dict[str, Any]] = {}

    async def _create_index(
        self, index_name: str, index_body: dict, force_recreate: bool = False
    ) -> bool:
        self.bodies[index_name] = index_body
        return True


@pytest.mark.asyncio
async def test_vehicle_and_face_indexes_use_their_own_hnsw_parameters() -> None:
    rec = _Recorder()
    await rec.create_vehicles_index()
    await rec.create_faces_index()
    veh, face = rec.bodies.values()
    for body in (veh, face):
        assert body['settings']['index'] == {
            'number_of_shards': 1,
            'number_of_replicas': 0,
            'knn': True,
        }
    vec = veh['mappings']['properties']['embedding']
    assert vec['type'] == 'knn_vector'
    assert vec['dimension'] == 512
    assert vec['method']['space_type'] == 'cosinesimil'
    assert vec['method']['parameters'] == {'ef_construction': 256, 'm': 12}
    assert face['mappings']['properties']['embedding']['method']['parameters'] == {
        'ef_construction': 1024,
        'm': 32,
    }
