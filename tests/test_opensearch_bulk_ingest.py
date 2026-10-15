"""``BulkMixin.bulk_ingest``: action construction, routing and failure semantics."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytest

from src.clients.opensearch import bulk as bulk_module
from src.clients.opensearch.bulk import BulkMixin


pytestmark = pytest.mark.asyncio


class Store(BulkMixin):
    def __init__(self) -> None:
        self.client = object()  # type: ignore[assignment]
        self.embedding_dim = 3


class BulkRecorder:
    def __init__(self, counts: dict[str, Any] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.counts = counts or {}

    async def __call__(
        self, client: Any, actions: list[dict[str, Any]], **kw: Any
    ) -> tuple[Any, list[Any]]:
        index = actions[0]['_index']
        self.calls.append({'client': client, 'actions': actions, 'kw': kw, 'index': index})
        value = self.counts.get(index, len(actions))
        if isinstance(value, Exception):
            raise value
        return value, []


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> BulkRecorder:
    rec = BulkRecorder()
    monkeypatch.setattr(bulk_module, 'async_bulk', rec)
    return rec


def _by_index(rec: BulkRecorder) -> dict[str, list[dict[str, Any]]]:
    return {c['index']: c['actions'] for c in rec.calls}


async def test_empty_input_makes_no_bulk_call(recorder: BulkRecorder) -> None:
    res = await Store().bulk_ingest([])
    assert res == {'global': 0, 'vehicles': 0, 'people': 0, 'errors': []}
    assert recorder.calls == []


async def test_global_document_shape(recorder: BulkRecorder) -> None:
    store = Store()
    res = await store.bulk_ingest(
        [
            {
                'image_id': 'a',
                'image_path': '/a',
                'global_embedding': np.array([1.0, 2.0]),
                'width': 0,
                'height': 5,
                'metadata': {'k': 1},
            },
            {'image_id': 'b', 'image_path': '/b', 'global_embedding': [3.0], 'metadata': {}},
        ],
        refresh=True,
    )
    assert res == {'global': 2, 'vehicles': 0, 'people': 0, 'errors': []}
    [call] = recorder.calls
    assert call['client'] is store.client
    assert call['kw'] == {'refresh': True, 'raise_on_error': False}
    first, second = call['actions']
    assert (first['_index'], first['_id']) == ('visual_search_global', 'a')
    src = first['_source']
    assert src['global_embedding'] == [1.0, 2.0]
    assert (src['width'], src['height'], src['metadata']) == (0, 5, {'k': 1})
    assert src['indexed_at']
    assert set(second['_source']) == {'image_id', 'image_path', 'global_embedding', 'indexed_at'}
    assert second['_source']['global_embedding'] == [3.0]


async def test_boxes_route_by_category_for_arrays_and_lists(recorder: BulkRecorder) -> None:
    docs = [
        {
            'image_id': 'a',
            'image_path': '/a',
            'global_embedding': [0.0],
            'box_embeddings': np.array([[0.1, 0], [0.2, 0], [0.3, 0]]),
            'det_classes': np.array([2, 0, 16]),
            'normalized_boxes': np.array([[0.0, 0.0, 1.0, 1.0]] * 3),
            'det_scores': np.array([0.9, 0.8, 0.7]),
            'metadata': {'m': 1},
        },
        {
            'image_id': 'b',
            'image_path': '/b',
            'global_embedding': [0.0],
            'box_embeddings': [[0.5, 0.5]],
            'det_classes': [0],
        },
    ]
    res = await Store().bulk_ingest(docs)
    assert res == {'global': 2, 'vehicles': 1, 'people': 2, 'errors': []}
    by = _by_index(recorder)
    [veh] = by['visual_search_vehicles']
    assert veh['_id'] == 'a_box_0'
    assert veh['_source']['class_name'] == 'car'
    assert veh['_source']['class_id'] == 2
    assert veh['_source']['embedding'] == [0.1, 0.0]
    assert veh['_source']['box'] == [0.0, 0.0, 1.0, 1.0]
    assert veh['_source']['confidence'] == pytest.approx(0.9)
    assert veh['_source']['metadata'] == {'m': 1}
    assert [a['_id'] for a in by['visual_search_people']] == ['a_box_1', 'b_box_0']
    person_a, person_b = (a['_source'] for a in by['visual_search_people'])
    assert person_a['has_face'] is False
    assert 'class_id' not in person_a
    assert person_a['confidence'] == pytest.approx(0.8)
    assert person_a['box'] == [0.0, 0.0, 1.0, 1.0]
    assert person_a['metadata'] == {'m': 1}
    assert person_b['embedding'] == [0.5, 0.5]
    assert not ({'box', 'confidence', 'metadata'} & set(person_b))


async def test_boxes_need_both_keys(recorder: BulkRecorder) -> None:
    base = {'image_id': 'a', 'image_path': '/a', 'global_embedding': [0.0]}
    await Store().bulk_ingest(
        [{**base, 'box_embeddings': [[1.0]]}, {**base, 'image_id': 'b', 'det_classes': [0]}]
    )
    assert [c['index'] for c in recorder.calls] == ['visual_search_global']


async def test_bulk_order_is_global_vehicles_people(recorder: BulkRecorder) -> None:
    await Store().bulk_ingest(
        [
            {
                'image_id': 'a',
                'image_path': '/a',
                'global_embedding': [0.0],
                'box_embeddings': [[1.0], [2.0]],
                'det_classes': [0, 2],
            }
        ]
    )
    assert [c['index'] for c in recorder.calls] == [
        'visual_search_global',
        'visual_search_vehicles',
        'visual_search_people',
    ]


async def test_reported_counts_come_from_the_bulk_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = BulkRecorder({'visual_search_global': 7, 'visual_search_vehicles': 3})
    monkeypatch.setattr(bulk_module, 'async_bulk', rec)
    res = await Store().bulk_ingest(
        [
            {
                'image_id': 'a',
                'image_path': '/a',
                'global_embedding': [0.0],
                'box_embeddings': [[1.0], [2.0]],
                'det_classes': [0, 2],
            }
        ]
    )
    assert res == {'global': 7, 'vehicles': 3, 'people': 1, 'errors': []}


async def test_failure_keeps_earlier_counts_and_stops(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    rec = BulkRecorder({'visual_search_vehicles': RuntimeError('vehicles down')})
    monkeypatch.setattr(bulk_module, 'async_bulk', rec)
    with caplog.at_level(logging.INFO, logger='src.clients.opensearch.bulk'):
        res = await Store().bulk_ingest(
            [
                {
                    'image_id': 'a',
                    'image_path': '/a',
                    'global_embedding': [0.0],
                    'box_embeddings': [[1.0], [2.0]],
                    'det_classes': [0, 2],
                }
            ]
        )
    assert res == {'global': 1, 'vehicles': 0, 'people': 0, 'errors': ['vehicles down']}
    assert [c['index'] for c in rec.calls] == ['visual_search_global', 'visual_search_vehicles']
    assert 'Bulk multi-index ingestion: global=1, vehicles=0, people=0' in [
        r.getMessage() for r in caplog.records
    ]
