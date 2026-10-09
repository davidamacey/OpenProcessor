"""A re-verify or a no-VLM accept never overwrites what a person stored, and
never leaves a stored ``proposed`` box unresolved (issue #57)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from _curation_app import mount_curation_routers
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.test_regions_router import _FakeRegionOS
from src.config import get_region_fields
from src.routers.curation import _raw_opensearch_dep, router as curation_router
from src.services.labeling.region_overlay import VlmBoxVerdict
from src.services.labeling.vlm_models import VlmCombinedReply

from .test_region_cascade_integrity import _drive_worker, _FakeOpenSearch
from .test_region_pending_verification_b1 import ITEM_BBOX, PROPOSED_BBOX, _seed_via_real_put_route
from .test_region_text_worker import _drive


if TYPE_CHECKING:
    from pathlib import Path


pytestmark = pytest.mark.usefixtures('reference_region_profile')

F = get_region_fields()
SECOND_BBOX = [0.5, 0.2, 0.8, 0.4]


def _accept(text: str) -> VlmCombinedReply:
    return VlmCombinedReply(
        img_id='c1',
        region_visible=True,
        region_boxes=[VlmBoxVerdict(box=1, bbox_correct=True, confidence='high', text_reply=text)],
    )


@pytest.mark.asyncio
async def test_reverify_keeps_cluster_text_and_provenance_and_files_no_human_chain_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeded = _seed_via_real_put_route()
    (b3,) = [b for b in seeded[F.boxes] if b['box_id'] == 'b3']
    b3.update(
        cluster_id=5,
        cluster_subid='5a',
        cluster_distance=0.12,
        detector_version=None,
        text='DNV20',
        text_source='human',
    )
    fake_os = _FakeOpenSearch({'c1': seeded}, search_delay=0.0, lag_searches=0)

    await _drive_worker(
        tmp_path,
        monkeypatch,
        fake_os=fake_os,
        primary=None,
        segmenter=None,
        reply=_accept('WRONG'),
    )

    doc = fake_os.live['c1']
    box = next(b for b in doc[F.boxes] if b['box_id'] == 'b3')
    assert box['state'] == 'accepted'
    assert (box['cluster_id'], box['cluster_subid'], box['cluster_distance']) == (5, '5a', 0.12)
    assert box['detector_version'] is None
    assert box['text'] == 'DNV20'
    assert box['text_source'] == 'human'
    chain = doc.get(F.detector_chain) or []
    assert not [e for e in chain if e.startswith('human:')], chain


@pytest.mark.asyncio
async def test_a_human_move_racing_a_vlm_verdict_leaves_no_verified_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeded = _seed_via_real_put_route()
    fake_os = _FakeOpenSearch({'c1': seeded}, search_delay=0.0, lag_searches=0)
    moved = [0.31, 0.61, 0.62, 0.77]
    calls: list[int] = []

    def human_moves_the_box(crops: list[Any], **_kw: Any) -> dict[str, Any]:
        if not calls:
            calls.append(1)
            for b in fake_os.live['c1'][F.boxes]:
                if b['box_id'] == 'b3':
                    b['bbox_norm'] = list(moved)
            fake_os.searchable = {k: dict(v) for k, v in fake_os.live.items()}
        return {c.crop_id: _accept('DNV20') for c in crops}

    await _drive_worker(
        tmp_path,
        monkeypatch,
        fake_os=fake_os,
        primary=None,
        segmenter=None,
        reply=_accept('DNV20'),
        combined_side_effect=human_moves_the_box,
        until_writes=1,
    )

    for _doc_id, written in fake_os.writes:
        b3 = next(b for b in written[F.boxes] if b['box_id'] == 'b3')
        assert b3['bbox_norm'] == moved
        if written.get(F.verified):
            assert b3['state'] == 'accepted', 'verified without an accepted box'


def _seed_two_proposed() -> dict[str, Any]:
    store = _FakeRegionOS(
        {
            'c1': {
                'crop_id': 'c1',
                'image_path': '/nonexistent/source.jpg',
                'bbox_norm': ITEM_BBOX,
                F.status: 'pending_detection',
                'class_name': 'sedan',
                'class_source': 'classifier_model',
                'confidence': 0.95,
                'created_at': '2026-09-24T00:00:00+00:00',
                F.revision: 1,
            }
        }
    )
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: store
    with TestClient(app) as client:
        resp = client.put(
            '/curation/projects/default/crops/c1/regions',
            json={
                'boxes': [
                    {'box_id': None, 'bbox_norm': PROPOSED_BBOX, 'state': 'proposed'},
                    {'box_id': None, 'bbox_norm': SECOND_BBOX, 'state': 'proposed'},
                ]
            },
        )
    assert resp.status_code == 200, resp.text
    doc = store._docs['c1']
    assert [b['state'] for b in doc[F.boxes]] == ['proposed', 'proposed']
    return doc


@pytest.mark.asyncio
async def test_no_vlm_accept_resolves_every_proposed_box(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_os = _FakeOpenSearch({'c1': _seed_two_proposed()}, search_delay=0.0, lag_searches=0)

    await _drive(
        tmp_path,
        monkeypatch,
        fake_os=fake_os,
        primary=None,
        segmenter=None,
        vlm_url='',
    )

    doc = fake_os.live['c1']
    assert [b['state'] for b in doc[F.boxes]] == ['accepted', 'accepted']
    assert doc[F.status] == 'detected'
    assert doc[F.count] == 2
