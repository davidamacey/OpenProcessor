"""``IngestMixin.ingest_image`` routing, documents, cluster assignment and error reporting."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.clients.opensearch.ingest import IngestMixin


pytestmark = pytest.mark.asyncio


class FakeClient:
    def __init__(self, fail_ids: set[str] | None = None) -> None:
        self.fail_ids = fail_ids or set()
        self.calls: list[dict[str, Any]] = []

    async def index(self, **kw: Any) -> None:
        if kw['id'] in self.fail_ids:
            msg = f'boom {kw["id"]}'
            raise RuntimeError(msg)
        self.calls.append(kw)


class Store(IngestMixin):
    def __init__(self, client: FakeClient) -> None:
        self.client = client  # type: ignore[assignment]
        self.embedding_dim = 3


class Assigner:
    def __init__(self, trained: set[str], raises: set[str] | None = None) -> None:
        self.trained = trained
        self.raises = raises or set()
        self.asked: list[tuple[str, list[float]]] = []

    def is_trained(self, idx: Any) -> bool:
        return str(getattr(idx, 'value', idx)) in self.trained

    def assign_cluster(self, idx: Any, emb: np.ndarray) -> Any:
        name = str(getattr(idx, 'value', idx))
        self.asked.append((name, emb.tolist()))
        if name in self.raises:
            msg = 'assign failed'
            raise RuntimeError(msg)
        return SimpleNamespace(cluster_id=len(self.asked), distance=0.5)


G = np.array([1.0, 0.0, 0.0])
BOXES = np.array([[0.1, 0, 0], [0, 0.2, 0], [0, 0, 0.3], [0.4, 0, 0]])
CLASSES = [2, 0, 16, 7]  # car, person, dog (skipped), truck
BOX_XY = np.array([[0.1, 0.1, 0.5, 0.5]] * 4)
SCORES = [0.9, 0.8, 0.7, 0.6]


def _by_index(client: FakeClient) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for c in client.calls:
        out.setdefault(c['index'], []).append(c)
    return out


async def test_global_only_minimal_doc() -> None:
    client = FakeClient()
    res = await Store(client).ingest_image('img', '/p.jpg', G)
    assert res == {
        'global': True,
        'vehicles': 0,
        'people': 0,
        'skipped': 0,
        'clustered': 0,
        'errors': [],
    }
    [call] = client.calls
    assert call['index'] == 'visual_search_global'
    assert call['id'] == 'img'
    assert call['refresh'] is False
    doc = call['body']
    assert set(doc) == {'image_id', 'image_path', 'global_embedding', 'indexed_at'}
    assert doc['global_embedding'] == [1.0, 0.0, 0.0]


async def test_global_optional_fields_only_when_truthy() -> None:
    client = FakeClient()
    await Store(client).ingest_image(
        'img',
        '/p.jpg',
        G,
        image_width=0,
        image_height=480,
        metadata={},
        imohash='',
        file_size_bytes=0,
    )
    doc = client.calls[0]['body']
    assert doc['height'] == 480
    assert not ({'width', 'metadata', 'imohash', 'file_size_bytes'} & set(doc))
    client = FakeClient()
    await Store(client).ingest_image(
        'img',
        '/p.jpg',
        G,
        image_width=640,
        image_height=480,
        metadata={'k': 1},
        imohash='abc',
        file_size_bytes=99,
    )
    doc = client.calls[0]['body']
    assert (doc['width'], doc['metadata'], doc['imohash'], doc['file_size_bytes']) == (
        640,
        {'k': 1},
        'abc',
        99,
    )


async def test_boxes_are_routed_by_category() -> None:
    client = FakeClient()
    res = await Store(client).ingest_image(
        'img',
        '/p.jpg',
        G,
        box_embeddings=BOXES,
        normalized_boxes=BOX_XY,
        det_classes=CLASSES,
        det_scores=SCORES,
        metadata={'m': 1},
    )
    assert (res['vehicles'], res['people'], res['skipped'], res['errors']) == (2, 1, 1, [])
    by = _by_index(client)
    veh = [c['body'] for c in by['visual_search_vehicles']]
    assert [c['id'] for c in by['visual_search_vehicles']] == ['img_box_0', 'img_box_3']
    assert veh[0]['detection_id'] == 'img_box_0'
    assert veh[0]['class_id'] == 2
    assert veh[0]['class_name'] == 'car'
    assert veh[1]['class_name'] == 'truck'
    assert veh[0]['embedding'] == [0.1, 0.0, 0.0]
    assert veh[0]['box'] == [0.1, 0.1, 0.5, 0.5]
    assert veh[0]['confidence'] == 0.9
    assert veh[0]['metadata'] == {'m': 1}
    assert veh[0]['image_id'] == 'img'
    assert veh[0]['image_path'] == '/p.jpg'
    [person_call] = by['visual_search_people']
    assert person_call['id'] == 'img_box_1'
    assert person_call['refresh'] is False
    person = person_call['body']
    assert person['has_face'] is False
    assert 'class_id' not in person
    assert person['confidence'] == 0.8
    assert person['box'] == [0.1, 0.1, 0.5, 0.5]
    assert person['metadata'] == {'m': 1}


async def test_box_optional_fields_omitted() -> None:
    client = FakeClient()
    await Store(client).ingest_image('img', '/p', G, box_embeddings=BOXES[:2], det_classes=[2, 0])
    for c in client.calls[1:]:
        assert not ({'box', 'confidence', 'metadata'} & set(c['body']))


async def test_boxes_need_both_embeddings_and_classes() -> None:
    client = FakeClient()
    await Store(client).ingest_image('img', '/p', G, box_embeddings=BOXES)
    await Store(client).ingest_image('img', '/p', G, det_classes=CLASSES)
    assert [c['index'] for c in client.calls] == ['visual_search_global'] * 2


async def test_clustering_assigns_trained_indexes_only() -> None:
    client = FakeClient()
    assigner = Assigner(trained={'global', 'people'})
    res = await Store(client).ingest_image(
        'img',
        '/p',
        G,
        box_embeddings=BOXES,
        det_classes=CLASSES,
        clustering_service=assigner,
    )
    assert res['clustered'] == 2
    assert [a[0] for a in assigner.asked] == ['global', 'people']
    by = _by_index(client)
    g = by['visual_search_global'][0]['body']
    assert (g['cluster_id'], g['cluster_distance']) == (1, 0.5)
    assert g['clustered_at'] == g['indexed_at']
    p = by['visual_search_people'][0]['body']
    assert p['cluster_id'] == 2
    assert p['clustered_at'] == p['indexed_at']
    assert not any('cluster_id' in c['body'] for c in by['visual_search_vehicles'])


async def test_cluster_failure_is_logged_and_does_not_block(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = FakeClient()
    assigner = Assigner(trained={'global', 'vehicles', 'people'}, raises={'global', 'people'})
    with caplog.at_level(logging.WARNING, logger='src.clients.opensearch.ingest'):
        res = await Store(client).ingest_image(
            'img',
            '/p',
            G,
            box_embeddings=BOXES,
            det_classes=CLASSES,
            clustering_service=assigner,
        )
    assert res['global'] is True
    assert res['clustered'] == 2  # the two vehicles
    assert res['errors'] == []
    messages = [r.getMessage() for r in caplog.records]
    assert 'Cluster assignment failed for global: assign failed' in messages
    assert 'Cluster assignment failed for person: assign failed' in messages
    assert not any('cluster_id' in c['body'] for c in client.calls if 'people' in c['index'])


async def test_index_errors_are_collected_per_document() -> None:
    client = FakeClient(fail_ids={'img', 'img_box_0'})
    res = await Store(client).ingest_image(
        'img', '/p', G, box_embeddings=BOXES, det_classes=CLASSES
    )
    assert res['global'] is False
    assert res['errors'] == ['Global: boom img', 'img_box_0: boom img_box_0']
    assert (res['vehicles'], res['people'], res['skipped']) == (1, 1, 1)


async def test_summary_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger='src.clients.opensearch.ingest'):
        await Store(FakeClient()).ingest_image(
            'img', '/p', G, box_embeddings=BOXES, det_classes=CLASSES
        )
    assert 'Multi-index ingestion: global=True, vehicles=2, people=1, skipped=1, clustered=0' in [
        r.getMessage() for r in caplog.records
    ]
