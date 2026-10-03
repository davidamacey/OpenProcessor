"""The small shared pieces the full-image pass stands on."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from src.clients.curation_opensearch import ClassRegistryError
from src.services.curation.class_ensure import ensure_class_by_name
from src.services.curation.ingest_class_sources import (
    OPEN_VOCAB_CLASS_SOURCE,
    unlabeled_proposal_class_sources,
)
from src.services.curation.open_vocab_fields import (
    OPEN_VOCAB_IMAGE_MAPPING,
    OPEN_VOCAB_ITEM_MAPPING,
    ensure_open_vocab_fields,
)


class _Entry:
    def __init__(self, class_id: int, name: str, deprecated: bool = False) -> None:
        self.class_id, self.class_name, self.deprecated = class_id, name, deprecated


class _Registry:
    def __init__(self, entries: list[_Entry]) -> None:
        self.entries = entries
        self.race: _Entry | None = None

    def load(self) -> Any:
        return type('F', (), {'classes': self.entries})()

    def add_class(self, name: str, group: str = 'unknown', notes: str = '') -> int:  # noqa: ARG002
        if self.race is not None:
            self.entries.append(self.race)
            raise ClassRegistryError('duplicate')
        self.entries.append(_Entry(len(self.entries), name))
        return len(self.entries) - 1


def test_ensure_class_matches_by_name_ignoring_case_and_deprecated() -> None:
    reg = _Registry([_Entry(0, 'Cone', deprecated=True), _Entry(1, 'cone')])
    assert ensure_class_by_name(reg, ' CONE ', group='g') == (1, 'cone')


def test_ensure_class_uses_the_one_name_equality_rule_and_returns_the_registry_spelling() -> None:
    reg = _Registry([_Entry(4, 'traffic_light')])
    assert ensure_class_by_name(reg, 'Traffic Light', group='g') == (4, 'traffic_light')
    assert len(reg.entries) == 1


def test_ensure_class_adds_a_missing_name_once() -> None:
    reg = _Registry([_Entry(0, 'cup')])
    assert ensure_class_by_name(reg, 'cone', group='g') == (1, 'cone')
    assert ensure_class_by_name(reg, 'cone', group='g') == (1, 'cone')
    assert len(reg.entries) == 2


def test_ensure_class_tolerates_a_concurrent_add_of_the_same_name() -> None:
    reg = _Registry([])
    reg.race = _Entry(7, 'cone')
    assert ensure_class_by_name(reg, 'cone', group='g') == (7, 'cone')


def test_ensure_class_refuses_a_name_with_nothing_slug_safe() -> None:
    with pytest.raises(ClassRegistryError):
        ensure_class_by_name(_Registry([]), ' !! ', group='g')


def test_ensure_class_reraises_when_the_failure_is_not_a_race() -> None:
    class Broken(_Registry):
        def add_class(self, name: str, group: str = '', notes: str = '') -> int:  # noqa: ARG002
            raise ClassRegistryError('class_name must be non-empty')

    with pytest.raises(ClassRegistryError):
        ensure_class_by_name(Broken([]), 'x', group='g')


def test_open_vocab_items_count_as_unlabeled_proposals() -> None:
    assert OPEN_VOCAB_CLASS_SOURCE in unlabeled_proposal_class_sources()


@pytest.mark.asyncio
async def test_mapping_migration_puts_every_field_on_the_bound_projects_indexes() -> None:
    from src.config import get_curation_config

    calls: list[tuple[str, str]] = []

    class Indices:
        async def put_mapping(self, index: str, body: dict[str, Any]) -> dict[str, bool]:
            ((field, _spec),) = body['properties'].items()
            calls.append((index, field))
            return {'acknowledged': True}

    client = type('C', (), {'indices': Indices()})()
    await ensure_open_vocab_fields(client)  # type: ignore[arg-type]

    cfg = get_curation_config()
    assert {f for i, f in calls if i == cfg.items_index} == set(OPEN_VOCAB_ITEM_MAPPING)
    assert {f for i, f in calls if i == cfg.images_index} == set(OPEN_VOCAB_IMAGE_MAPPING)


@pytest.mark.asyncio
async def test_mapping_migration_surfaces_a_real_failure() -> None:
    class Indices:
        async def put_mapping(self, **_kw: Any) -> None:
            raise RuntimeError('connection refused')

    client = type('C', (), {'indices': Indices()})()
    with pytest.raises(RuntimeError, match='connection refused'):
        await ensure_open_vocab_fields(client)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_segment_image_http_sends_the_floor_and_maps_polygons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.services.detection import segmenter_http

    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                'candidates': [
                    {
                        'bbox_norm': [0.1, 0.1, 0.4, 0.4],
                        'score': 0.9,
                        'mask_iou': 0.7,
                        'mask_polygon': [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4]],
                    }
                ]
            },
        )

    real = httpx.AsyncClient
    monkeypatch.setattr(
        segmenter_http.httpx,
        'AsyncClient',
        lambda **kw: real(transport=httpx.MockTransport(handler), **kw),
    )
    monkeypatch.setenv('OP_SEGMENTER_URL', 'http://seg.invalid:8000')

    out = await segmenter_http.segment_image_http(
        b'JPEG', 'cup', min_score=0.3, max_candidates=5, return_masks=True
    )

    assert seen[0]['min_score'] == 0.3
    assert seen[0]['return_masks'] is True
    assert out[0].mask_polygon == ((0.1, 0.1), (0.4, 0.1), (0.4, 0.4))
    assert out[0].rectangularity == 0.7
    assert out[0].source == 'sam3'


@pytest.mark.asyncio
async def test_segment_image_http_without_a_segmenter_is_an_outage_not_no_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.services.detection.segmenter_http import SegmenterCallError, segment_image_http

    monkeypatch.delenv('OP_SEGMENTER_URL', raising=False)
    with pytest.raises(SegmenterCallError):
        await segment_image_http(b'J', 'cup', min_score=None, max_candidates=1, return_masks=False)
