"""A box says the same thing on the gallery card and in the image detail.

`GET /images` draws each card's boxes from a `top_hits` that fetches only some fields of the
items, `GET /images/{id}` from the whole documents. A name field the card did not fetch made every
detector-only box read "unlabeled" in the gallery and "dog" in the detail. The fetched fields
are derived from what the box serializer reads, so a new input cannot be forgotten again.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch
from src.config import IndexRole, index_name
from src.config.curation_accessors import base_curation_config
from src.routers.curation import images_view as route
from src.services.curation.image_boxes import box_source_fields, item_box_views


if TYPE_CHECKING:
    import pytest

CFG = base_curation_config()
ITEMS = CFG.items_index
IMAGES = index_name(CFG, IndexRole.IMAGES)
BASE = '/curation/projects/default'


def _client(fake: Any, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from _curation_app import mount_curation_routers

    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    async def _policy() -> str:
        return 'store'

    monkeypatch.setattr(route, 'get_location_policy', _policy)
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)


def _item(crop_id: str, rank: int, **extra: Any) -> dict[str, Any]:
    return {
        'crop_id': crop_id,
        'image_id': 'img-a',
        'crop_rank_in_image': rank,
        'bbox_norm': [0.1, 0.1, 0.5, 0.5],
        **extra,
    }


# One item per way the serializer can name a box.
_DOCS = {
    'cls': _item('cls', 1, class_id=3, class_name='sedan'),
    'det': _item('det', 2, proposal_name='dog', detector_class_name='dog'),
    'det_only': _item('det_only', 3, detector_class_name='cat'),
    'vlm': _item(
        'vlm',
        4,
        class_source='vlm_new_class_pending',
        vlm_proposed_class='zebra',
        vlm_raw_class='z',
    ),
    'none': _item('none', 5),
}


def _fake() -> QueryFakeOpenSearch:
    return QueryFakeOpenSearch(
        {
            ITEMS: {k: dict(v) for k, v in _DOCS.items()},
            IMAGES: {
                'img-a': {
                    'image_id': 'img-a',
                    'image_path': '/data/img-a.jpg',
                    'width': 640,
                    'height': 480,
                    'source': 'disk',
                    'indexed_at': '2026-09-03T00:00:00',
                }
            },
        }
    )


def _names(boxes: list[dict[str, Any]]) -> dict[str, tuple[str, str]]:
    return {b['crop_id']: (b['label'], b['label_kind']) for b in boxes if b['kind'] == 'item'}


def test_gallery_and_detail_name_a_box_the_same(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(_fake(), monkeypatch)
    card = client.get(f'{BASE}/images').json()['items'][0]
    detail = client.get(f'{BASE}/images/img-a').json()
    assert _names(card['boxes']) == _names(detail['boxes'])
    assert _names(card['boxes'])['det'] == ('dog', 'detector')
    assert _names(card['boxes'])['det_only'] == ('cat', 'detector')
    assert _names(card['boxes'])['none'] == ('unlabeled', 'none')


class _Reads(dict[str, Any]):
    """A stored item that records every field the serializer asks of it."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.read: set[str] = set()

    def get(self, key: str, default: Any = None) -> Any:
        self.read.add(key)
        return super().get(key, default)

    def __getitem__(self, key: str) -> Any:
        self.read.add(key)
        return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        self.read.add(str(key))
        return super().__contains__(key)


def test_the_card_fetches_every_field_the_box_serializer_reads() -> None:
    read: set[str] = set()
    for doc in [*_DOCS.values(), {**_DOCS['cls'], 'class_validated': True, 'class_source': 'vlm'}]:
        src = _Reads(doc)
        item_box_views(src, doc['crop_id'])
        read |= src.read
    fetched = set(box_source_fields())
    assert read <= fetched, sorted(read - fetched)


def test_the_card_draws_the_region_boxes_the_detail_draws(monkeypatch: pytest.MonkeyPatch) -> None:
    """The card's top_hits fetches the region box list (the lock fields carry it), so a card draws
    every region box the detail draws."""
    fake = _fake()
    fake.docs(ITEMS)['cls']['region_boxes'] = [
        {'box_id': 'r1', 'bbox_norm': [0.2, 0.2, 0.3, 0.3], 'state': 'accepted'},
        {'box_id': 'r2', 'bbox_norm': [0.6, 0.6, 0.7, 0.7], 'state': 'rejected'},
    ]
    client = _client(fake, monkeypatch)
    card = client.get(f'{BASE}/images').json()['items'][0]
    detail = client.get(f'{BASE}/images/img-a').json()

    def keyed(boxes: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
        return sorted(
            (b['crop_id'], b.get('box_id') or '', b['kind'], b['tone'], b['label']) for b in boxes
        )

    assert [b['box_id'] for b in detail['boxes'] if b['kind'] == 'region'] == ['r1', 'r2']
    assert keyed(card['boxes']) == keyed(detail['boxes'])


def test_a_held_out_item_is_counted_not_listed_and_only_the_detail_draws_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one difference between a card and the detail (walk "15 of 30 images draw fewer boxes"):
    the test holdout is not listed on a card, and the card says how many it left out."""
    fake = _fake()
    fake.docs(ITEMS)['det'].update(
        test_holdout=True,
        region_boxes=[{'box_id': 'h1', 'bbox_norm': [0.2, 0.2, 0.3, 0.3], 'state': 'accepted'}],
    )
    client = _client(fake, monkeypatch)
    card = client.get(f'{BASE}/images').json()['items'][0]
    detail = client.get(f'{BASE}/images/img-a').json()
    assert 'det' not in {b['crop_id'] for b in card['boxes']}
    assert 'det' in {b['crop_id'] for b in detail['boxes']}
    assert card['counts']['in_holdout'] == detail['counts']['in_holdout'] == 1
