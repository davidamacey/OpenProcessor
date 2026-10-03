"""A human box edit keeps per-box vectors in step: new, moved and newly
accepted boxes are embedded; a failure leaves them pending, never fails the edit."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch
from curation.reprocess_fixtures import (
    F,
    FakePE,
    box,
    docs,
    images_index,
    item,
    items_index,
    jpeg_bytes,
    servable_root,
)
from curation.test_regions_router import _FakeRegionOS
from src.services.curation.region_box_embeddings import current_vectors, entry_for
from src.services.curation.region_box_refresh import refresh_box_embeddings
from src.services.curation.region_boxes import read_boxes


if TYPE_CHECKING:
    from pathlib import Path


def _world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **extra: Any) -> QueryFakeOpenSearch:
    root = servable_root(tmp_path, monkeypatch)
    path = root / 'a.jpg'
    path.write_bytes(jpeg_bytes())
    boxes = (
        box('kept', state='accepted', detector='d', reason=None, bbox=(0.1, 0.1, 0.3, 0.3)),
        box('new', state='accepted', detector='human', reason=None, bbox=(0.5, 0.5, 0.8, 0.8)),
        box('moved', state='accepted', detector='d', reason=None, bbox=(0.6, 0.1, 0.9, 0.3)),
        box('proposed', state='proposed', detector='d', reason=None, bbox=(0.2, 0.6, 0.4, 0.9)),
    )
    stale_moved = box('moved', bbox=(0.0, 0.1, 0.2, 0.3))
    from src.services.curation.region_boxes import RegionBox

    def entry(b: dict[str, Any], bbox: Any = None) -> dict[str, Any]:
        rb = RegionBox(
            box_id=b['box_id'], bbox_norm=tuple(bbox or b['bbox_norm']), state=b['state']
        )
        return entry_for(rb, [7.0, 7.0, 7.0])

    doc = item(
        'c1',
        image_path=str(path),
        boxes=boxes,
        **{
            F.box_embeddings: [
                entry(boxes[0]),  # kept: valid
                entry(boxes[2], stale_moved['bbox_norm']),  # moved: computed from the old geometry
            ]
        },
        **extra,
    )
    return QueryFakeOpenSearch(
        {
            items_index(): {'c1': doc},
            images_index(): {'img-1': {'image_id': 'img-1', 'image_path': str(path)}},
        }
    )


@pytest.mark.asyncio
async def test_new_and_moved_boxes_are_embedded_and_valid_ones_left_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _world(tmp_path, monkeypatch)
    pe = FakePE()
    result = await refresh_box_embeddings(fake, ['c1'], pe=pe)

    assert result == {'embedded': 2, 'pending': 0}
    assert pe.crop_calls == 2  # new + moved; kept has a valid vector, proposed is not embeddable
    vectors = current_vectors(docs(fake)['c1'])
    assert vectors['kept'] == [7.0, 7.0, 7.0]  # untouched
    assert vectors['new'] == pytest.approx([0.0, 0.0, 1.0])
    assert vectors['moved'] == pytest.approx([0.0, 0.0, 1.0])
    assert 'proposed' not in vectors


@pytest.mark.asyncio
async def test_a_deleted_box_leaves_no_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _world(tmp_path, monkeypatch)
    doc = docs(fake)['c1']
    doc[F.box_embeddings].append(
        {'box_id': 'gone', 'bbox_norm': [0.1, 0.1, 0.2, 0.2], 'embedding': [1.0, 1.0, 1.0]}
    )
    await refresh_box_embeddings(fake, ['c1'], pe=FakePE())
    assert 'gone' not in {e['box_id'] for e in docs(fake)['c1'][F.box_embeddings]}


@pytest.mark.asyncio
async def test_an_encoder_failure_leaves_the_boxes_pending_and_raises_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Broken(FakePE):
        async def embed_crops(self, crops: list[Any], max_batch: int = 32) -> Any:  # noqa: ARG002
            raise RuntimeError('triton down')

    fake = _world(tmp_path, monkeypatch)
    result = await refresh_box_embeddings(fake, ['c1'], pe=Broken())
    assert result == {'embedded': 0, 'pending': 2}
    assert 'new' not in current_vectors(docs(fake)['c1'])


@pytest.mark.asyncio
async def test_without_an_encoder_the_boxes_are_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr('src.services.curation.region_box_refresh._encoder', lambda: None)
    fake = _world(tmp_path, monkeypatch)
    assert await refresh_box_embeddings(fake, ['c1']) == {'embedded': 0, 'pending': 2}


@pytest.mark.asyncio
async def test_nothing_missing_means_no_encoder_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _world(tmp_path, monkeypatch)
    pe = FakePE()
    await refresh_box_embeddings(fake, ['c1'], pe=pe)
    pe.crop_calls = 0
    assert await refresh_box_embeddings(fake, ['c1'], pe=pe) == {'embedded': 0, 'pending': 0}
    assert pe.crop_calls == 0
    assert read_boxes(docs(fake)['c1'], F)  # the box list itself was never rewritten


@pytest.mark.usefixtures('reference_region_profile')
def test_every_human_box_edit_route_refreshes_the_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    async def spy(_client: Any, crop_ids: Any, **_: Any) -> dict[str, int]:
        calls.append(list(crop_ids))
        return {'embedded': 0, 'pending': 0}

    for module in ('regions_boxes_edit', 'regions_edit'):
        monkeypatch.setattr(f'src.routers.curation.{module}.refresh_box_embeddings', spy)
    doc = {
        'crop_id': 'crop-1',
        F.status: 'detected',
        F.boxes: [
            {'box_id': 'b1', 'bbox_norm': [0.1, 0.1, 0.2, 0.2], 'state': 'accepted', 'score': 0.9}
        ],
        F.box_seq: 1,
        F.revision: 1,
        F.count: 1,
        F.rejected_count: 0,
    }
    fake = _FakeRegionOS({'crop-1': doc})
    from _curation_app import mount_curation_routers

    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    base = '/curation/projects/default'
    with TestClient(app) as client:
        responses = [
            client.put(f'{base}/crops/crop-1/regions', json={'boxes': [{'box_id': 'b1'}]}),
            client.patch(f'{base}/crops/crop-1/regions/b1', json={'state': 'rejected'}),
            client.post(
                f'{base}/regions/batch_box_state',
                json={'targets': [{'crop_id': 'crop-1', 'box_id': 'b1'}], 'state': 'accepted'},
            ),
            client.patch(f'{base}/crops/crop-1/region_meta', json={'region_status': 'detected'}),
            client.post(
                f'{base}/regions/batch_status',
                json={'crop_ids': ['crop-1'], 'region_status': 'detected'},
            ),
            client.put(
                f'{base}/crops/batch_regions',
                json={
                    'crop_ids': ['crop-1'],
                    'boxes': [{'box_id': None, 'bbox_norm': [0.5, 0.5, 0.6, 0.6]}],
                },
            ),
        ]
    assert [r.status_code for r in responses] == [200] * 6, [r.text for r in responses]
    assert len(calls) == 6
    assert all(r.json()['vector_refresh'] == {'embedded': 0, 'pending': 0} for r in responses)
