"""`PATCH /crops/{id}/geometry` and `POST /crops/{id}/geometry/undo`.

A box edit is saved at once, checked against the revision inside the OCC merger (so two editors
cannot both win), drops the old vector before the re-embed (a failed embed never leaves it on a
moved box), refuses to strand region boxes, and is undone step by step with the pairing rule of
the other edit histories.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.reprocess_fixtures import FakePE, box, item, jpeg_bytes, make_fake, servable_root
from src.config import IndexRole, get_region_fields, index_name
from src.config.curation_accessors import base_curation_config
from src.services.curation import item_geometry as geometry_service
from src.services.curation.source_image_cache import crop_cache_file


if TYPE_CHECKING:
    from pathlib import Path

    from curation.query_fakes import QueryFakeOpenSearch

CFG = base_curation_config()
ITEMS = CFG.items_index
IMAGES = index_name(CFG, IndexRole.IMAGES)
BASE = '/curation/projects/default'
F = get_region_fields()
ORIGINAL = [0.1, 0.1, 0.5, 0.5]
MOVED = [0.2, 0.2, 0.7, 0.8]
FAR = [0.3, 0.3, 0.9, 0.9]


class _BrokenPE(FakePE):
    async def embed_crops(self, crops: list[Any], max_batch: int = 32) -> Any:
        raise RuntimeError('encoder down')


class _WatchingPE(FakePE):
    """Records whether the old vector was still stored when the encoder ran."""

    def __init__(self, fake: Any) -> None:
        super().__init__()
        self.fake = fake
        self.vector_at_embed: list[Any] = []

    async def embed_crops(self, crops: list[Any], max_batch: int = 32) -> Any:
        self.vector_at_embed.append(self.fake.docs(ITEMS)['it-1'].get('pe_embedding'))
        return await super().embed_crops(crops, max_batch)


def _client(fake: Any, pe: Any) -> TestClient:
    from _curation_app import mount_curation_routers

    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    app.state.pe_encoder = pe
    return TestClient(app)


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> QueryFakeOpenSearch:
    root = servable_root(tmp_path, monkeypatch)
    path = root / 'a.jpg'
    path.write_bytes(jpeg_bytes())
    machine = item(
        'it-1',
        image_id='img-1',
        image_path=str(path),
        bbox_norm=tuple(ORIGINAL),
        embedding_state='embedded',
        pe_embedding=[9.0, 9.0, 9.0],
        crop_area_norm=0.16,
        crop_rank_in_image=2,
    )
    other = item(
        'it-2',
        image_id='img-1',
        image_path=str(path),
        bbox_norm=(0.0, 0.0, 0.3, 0.3),
        crop_rank_in_image=3,
    )
    return make_fake(
        [machine, other],
        [{'image_id': 'img-1', 'image_path': str(path), 'width': 400, 'height': 300}],
    )


def _patch(client: TestClient, bbox: list[float], rev: int, **extra: Any) -> Any:
    return client.patch(
        f'{BASE}/crops/it-1/geometry',
        json={'bbox_norm': bbox, 'expected_geometry_revision': rev, **extra},
    )


def test_move_bumps_revision_and_keeps_detector_box(world: Any) -> None:
    client = _client(world, FakePE())
    first = _patch(client, MOVED, 0)
    assert first.status_code == 200, first.text
    doc = world.docs(ITEMS)['it-1']
    assert (doc['bbox_norm'], doc['geometry_revision'], doc['geometry_source']) == (
        MOVED,
        1,
        'human',
    )
    assert doc['detector_bbox_norm'] == ORIGINAL
    assert doc['crop_area_norm'] == pytest.approx(0.3)
    assert doc['crop_id'] == 'it-1'  # the id is the identity; the box is not part of it
    assert _patch(client, FAR, 1).status_code == 200
    again = world.docs(ITEMS)['it-1']
    assert (again['geometry_revision'], again['detector_bbox_norm']) == (2, ORIGINAL)
    assert again['bbox_norm'] == FAR


def test_ranks_follow_the_new_area(world: Any) -> None:
    _patch(_client(world, FakePE()), FAR, 0)
    assert [world.docs(ITEMS)[i]['crop_rank_in_image'] for i in ('it-1', 'it-2')] == [1, 2]


def test_stale_revision_is_409_with_the_current_item(world: Any) -> None:
    client = _client(world, FakePE())
    assert _patch(client, MOVED, 0).status_code == 200
    resp = _patch(client, FAR, 0)
    assert resp.status_code == 409
    detail = resp.json()['detail']
    assert detail['error'] == 'geometry_conflict'
    assert detail['current_geometry_revision'] == 1
    assert detail['item']['bbox_norm'] == MOVED
    assert world.docs(ITEMS)['it-1']['bbox_norm'] == MOVED


def test_two_editors_with_the_same_revision_one_wins(world: Any) -> None:
    """Both read revision 0; the conditional write lets one through and the loser's retry sees 1."""
    from src.core.exceptions import RequestRejectedError
    from src.services.curation.item_geometry import edit_geometry

    async def edit(bbox: list[float]) -> str:
        try:
            await edit_geometry(
                world, FakePE(), 'it-1', bbox=bbox, expected_revision=0, drop_region_box_ids=[]
            )
        except RequestRejectedError as exc:
            return 'conflict' if exc.status_code == 409 else 'error'
        return 'ok'

    real_get = world.get

    async def get_then_yield(**kw: Any) -> Any:
        out = await real_get(**kw)
        await asyncio.sleep(0)  # let the other editor read the same revision
        return out

    world.get = get_then_yield

    async def both() -> list[str]:
        return list(await asyncio.gather(edit(MOVED), edit(FAR)))

    assert sorted(asyncio.run(both())) == ['conflict', 'ok']
    assert world.docs(ITEMS)['it-1']['geometry_revision'] == 1


def test_move_drops_the_vector_and_re_embeds(world: Any) -> None:
    pe = _WatchingPE(world)
    assert _patch(_client(world, pe), MOVED, 0).json()['embedding_state'] == 'embedded'
    assert pe.vector_at_embed == [None]  # the old vector was gone before the encoder ran
    assert world.docs(ITEMS)['it-1']['pe_embedding'] == [0.0, 0.0, 1.0]


def test_a_failed_embed_leaves_no_vector_on_a_moved_box(world: Any) -> None:
    resp = _patch(_client(world, _BrokenPE()), MOVED, 0)
    assert resp.status_code == 200
    assert resp.json()['embedding_state'] == 'deferred'
    doc = world.docs(ITEMS)['it-1']
    assert doc.get('pe_embedding') is None
    assert doc['embedding_state'] == 'deferred'
    assert doc['bbox_norm'] == MOVED


def test_crop_cache_file_is_evicted(
    world: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / 'cache'
    cached = crop_cache_file(cache, 'it-1')
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b'old crop')
    monkeypatch.setattr(
        geometry_service,
        'get_curation_config',
        lambda: SimpleNamespace(items_index=ITEMS, crop_cache_dir=str(cache)),
    )
    assert _patch(_client(world, FakePE()), MOVED, 0).status_code == 200
    assert not cached.exists()


def test_regions_outside_are_refused_unless_dropped(world: Any) -> None:
    inside = box('b1', state='accepted', bbox=(0.15, 0.15, 0.3, 0.3))
    outside = box('b2', state='accepted', bbox=(0.4, 0.4, 0.5, 0.5))
    world.docs(ITEMS)['it-1'].update({F.boxes: [inside, outside], F.count: 2, F.rejected_count: 0})
    client = _client(world, FakePE())
    shrink = [0.1, 0.1, 0.35, 0.35]
    refused = _patch(client, shrink, 0)
    assert refused.status_code == 422
    assert refused.json()['detail']['error'] == 'regions_outside_item'
    assert refused.json()['detail']['box_ids'] == ['b2']
    assert world.docs(ITEMS)['it-1']['bbox_norm'] == ORIGINAL

    done = _patch(client, shrink, 0, drop_region_box_ids=['b2'])
    assert done.status_code == 200, done.text
    stored = world.docs(ITEMS)['it-1']
    assert [b['box_id'] for b in stored[F.boxes]] == ['b1']
    assert stored[F.count] == 1


@pytest.mark.parametrize(
    ('bbox', 'code'),
    [
        ([0.2, 0.2, 1.4, 0.7], 'box_out_of_image'),
        ([0.5, 0.2, 0.2, 0.7], 'degenerate_box'),
        ([0.2, 0.2, 0.201, 0.7], 'box_too_small'),
    ],
)
def test_box_refusals_apply_to_a_move(world: Any, bbox: list[float], code: str) -> None:
    resp = _patch(_client(world, FakePE()), bbox, 0)
    assert (resp.status_code, resp.json()['detail']['error']) == (422, code)
    assert world.docs(ITEMS)['it-1']['bbox_norm'] == ORIGINAL


def test_unknown_item_is_404(world: Any) -> None:
    resp = _client(world, FakePE()).patch(
        f'{BASE}/crops/missing/geometry',
        json={'bbox_norm': MOVED, 'expected_geometry_revision': 0},
    )
    assert resp.status_code == 404


def _undo(client: TestClient, rev: int) -> Any:
    return client.post(f'{BASE}/crops/it-1/geometry/undo', json={'expected_geometry_revision': rev})


def test_undo_restores_the_previous_geometry(world: Any) -> None:
    client = _client(world, FakePE())
    _patch(client, MOVED, 0)
    _patch(client, FAR, 1)
    back = _undo(client, 2)
    assert back.status_code == 200, back.text
    doc = world.docs(ITEMS)['it-1']
    assert doc['bbox_norm'] == MOVED
    assert doc['geometry_revision'] == 3  # an undo is a change a stale editor must see
    assert doc['pe_embedding'] == [0.0, 0.0, 1.0]  # re-embedded for the restored box
    assert _undo(client, 3).status_code == 200
    first = world.docs(ITEMS)['it-1']
    assert first['bbox_norm'] == ORIGINAL
    assert first.get('geometry_source') is None  # the machine box again


def test_undo_with_nothing_left_is_409(world: Any) -> None:
    client = _client(world, FakePE())
    assert _undo(client, 0).status_code == 409
    assert _undo(client, 0).json()['detail']['error'] == 'nothing_to_undo'
    _patch(client, MOVED, 0)
    assert _undo(client, 1).status_code == 200
    assert _undo(client, 2).status_code == 409  # an undo of an undo is not restorable


def test_a_stale_editor_after_an_undo_conflicts(world: Any) -> None:
    client = _client(world, FakePE())
    _patch(client, MOVED, 0)
    _undo(client, 1)
    assert _patch(client, FAR, 1).status_code == 409


def _two_regions(world: Any) -> None:
    inside = box('b1', state='accepted', bbox=(0.15, 0.15, 0.3, 0.3))
    outside = box('b2', state='accepted', bbox=(0.4, 0.4, 0.5, 0.5))
    world.docs(ITEMS)['it-1'].update({F.boxes: [inside, outside], F.count: 2, F.rejected_count: 0})


def test_undo_puts_back_the_region_boxes_a_move_dropped(world: Any) -> None:
    _two_regions(world)
    client = _client(world, FakePE())
    shrink = [0.1, 0.1, 0.35, 0.35]
    assert _patch(client, shrink, 0, drop_region_box_ids=['b2']).status_code == 200
    assert _undo(client, 1).status_code == 200
    stored = world.docs(ITEMS)['it-1']
    assert stored['bbox_norm'] == ORIGINAL
    assert [b['box_id'] for b in stored[F.boxes]] == ['b1', 'b2']
    assert stored[F.count] == 2


def test_undo_of_a_move_that_dropped_nothing_leaves_later_region_edits(world: Any) -> None:
    _two_regions(world)
    client = _client(world, FakePE())
    assert _patch(client, [0.05, 0.05, 0.6, 0.6], 0).status_code == 200
    world.docs(ITEMS)['it-1'][F.boxes] = [box('b1', state='rejected', bbox=(0.15, 0.15, 0.3, 0.3))]
    assert _undo(client, 1).status_code == 200
    assert [b['state'] for b in world.docs(ITEMS)['it-1'][F.boxes]] == ['rejected']


def test_undo_after_a_region_edit_that_followed_a_drop_is_refused(world: Any) -> None:
    _two_regions(world)
    client = _client(world, FakePE())
    assert _patch(client, [0.1, 0.1, 0.35, 0.35], 0, drop_region_box_ids=['b2']).status_code == 200
    world.docs(ITEMS)['it-1'][F.revision] = int(world.docs(ITEMS)['it-1'][F.revision]) + 1
    resp = _undo(client, 1)
    assert (resp.status_code, resp.json()['detail']['error']) == (409, 'regions_changed')
    assert world.docs(ITEMS)['it-1']['bbox_norm'] == [0.1, 0.1, 0.35, 0.35]


def test_undo_by_an_editor_holding_a_stale_revision_conflicts(world: Any) -> None:
    """A and B both hold revision 0; B moves; A's undo must not undo B's move."""
    client = _client(world, FakePE())
    _patch(client, MOVED, 0)  # an earlier edit A saw and may undo
    _patch(client, FAR, 1)  # B's move, revision 2
    resp = _undo(client, 1)
    assert resp.status_code == 409
    detail = resp.json()['detail']
    assert (detail['error'], detail['current_geometry_revision']) == ('geometry_conflict', 2)
    assert detail['item']['bbox_norm'] == FAR
    assert world.docs(ITEMS)['it-1']['bbox_norm'] == FAR


def test_moving_a_human_deleted_box_keeps_the_detector_anchor(world: Any) -> None:
    """A re-detect matches the detector's own box; without the anchor it would re-create it."""
    world.docs(ITEMS)['it-1'].update(
        {'class_excluded': True, 'excluded_by': 'human', 'excluded_reason': 'not_an_object'}
    )
    assert _patch(_client(world, FakePE()), MOVED, 0).status_code == 200
    assert world.docs(ITEMS)['it-1']['detector_bbox_norm'] == ORIGINAL


def test_a_crop_read_after_the_move_never_writes_the_cache_back(
    world: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only ingest writes crops into the cache, and it skips a geometry-locked item; a read between
    the move and the eviction can therefore not put the old crop back under the same id."""
    from src.services.curation import crop_bytes

    cache = tmp_path / 'cache'
    cached = crop_cache_file(cache, 'it-1')
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b'old crop')
    cfg = SimpleNamespace(items_index=ITEMS, crop_cache_dir=str(cache), crop_max_side=0)
    monkeypatch.setattr(geometry_service, 'get_curation_config', lambda: cfg)
    monkeypatch.setattr(crop_bytes, 'get_curation_config', lambda: cfg)
    assert _patch(_client(world, FakePE()), MOVED, 0).status_code == 200
    path = world.docs(ITEMS)['it-1']['image_path']
    fresh = crop_bytes.load_item_crop_jpeg('it-1', path, tuple(MOVED))  # type: ignore[arg-type]
    assert fresh is not None
    assert fresh != b'old crop'
    assert not cached.exists()
