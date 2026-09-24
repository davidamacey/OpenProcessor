"""F-26 (vlm.py:469 row): ``POST /vlm/verify_regions`` must not clobber a
concurrent human write.

The endpoint fetches each crop, calls the VLM to verify its region, then
used to bulk-write the verdict with a plain, unconditional ``_bulk`` (no
``if_seq_no``/``if_primary_term``). If a human verified/labeled the same
crop while the VLM round-trip was in flight, the VLM's write would
silently win. Fixed by routing the write through
``src.clients.occ.occ_skip_on_conflict_bulk`` (conditional bulk; a 409
conflict for a given id is skipped, not retried) with a merger that
re-checks the freshest ``current`` doc state for human ownership --
exactly the pattern ``vlm_label_batch`` already uses for class writes
(``_class_locked``) and the clustering orchestrator's bulk writers use for
cluster writes (F-3) -- since a human write landing mid-VLM-round-trip
lands well before occ_skip_on_conflict_bulk's own (much narrower)
mget-to-bulk OCC window.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from curation.query_fakes import QueryFakeOpenSearch
from src.config import get_curation_config, get_region_fields


ITEMS = get_curation_config().items_index
F = get_region_fields()


def _item(crop_id: str, **extra: Any) -> dict[str, Any]:
    return {
        'crop_id': crop_id,
        'image_path': f'/data/{crop_id}.jpg',
        F.bbox_norm: [0.1, 0.1, 0.5, 0.5],
        F.verified: None,
        F.reason: None,
        **extra,
    }


class _FakeVerdict:
    def __init__(self, is_region: bool = True, reason: str = 'looks real') -> None:
        self.is_region = is_region
        self.reason = reason


class _Labeler:
    async def verify_region(self, _crop: Any) -> _FakeVerdict:
        return _FakeVerdict()


class _RacingLabeler:
    """Human-verifies 'raced' mid-loop -- simulating a human write landing
    on OpenSearch between this endpoint's initial per-crop ``get`` (used
    to read region_box/image_path) and its final bulk write, which is
    exactly the round-trip window a real VLM call spans."""

    def __init__(self, fake: QueryFakeOpenSearch) -> None:
        self._fake = fake

    async def verify_region(self, crop: Any) -> _FakeVerdict:
        if crop.crop_id == 'raced':
            self._fake.docs(ITEMS)['raced'].update(
                **{F.verifier: 'human', F.verified: False, F.reason: 'not a real region'}
            )
        return _FakeVerdict()


def _setup(monkeypatch: pytest.MonkeyPatch, labeler: Any) -> None:
    import src.routers.curation.vlm as vlm_mod

    async def _no_pack(_os: Any) -> None:
        return None

    monkeypatch.setattr(vlm_mod, '_default_pack_name', _no_pack)
    monkeypatch.setattr(vlm_mod, '_get_vlm_labeler', lambda *_a, **_k: labeler)
    monkeypatch.setattr(vlm_mod, 'resolve_crop_root', lambda _path: SimpleNamespace())
    monkeypatch.setattr(vlm_mod, 'resolve_safe_path', lambda path, _root: path)
    monkeypatch.setattr(vlm_mod.THUMBNAIL_CACHE, 'get_or_compute', lambda *_a, **_k: b'jpeg-bytes')


@pytest.mark.asyncio
async def test_verify_regions_never_overwrites_a_doc_a_human_verified_mid_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.routers.curation.vlm as vlm_mod
    from src.routers.curation.vlm import VlmVerifyRegionsRequest

    fake = QueryFakeOpenSearch(
        {
            ITEMS: {
                'raced': _item('raced'),
                'plain': _item('plain'),
            }
        }
    )
    _setup(monkeypatch, _RacingLabeler(fake))

    resp = await vlm_mod.vlm_verify_regions(
        VlmVerifyRegionsRequest(crop_ids=['raced', 'plain']), fake
    )

    docs = fake.docs(ITEMS)
    # 'raced' lost the race -- the human's verdict (set mid-flight) must
    # survive; the VLM's write must not have landed on top of it.
    assert docs['raced'][F.verified] is False
    assert docs['raced'][F.reason] == 'not a real region'
    assert docs['raced'][F.verifier] == 'human'
    # 'plain' had no conflict -- the VLM write applies normally.
    assert docs['plain'][F.verified] is True
    assert docs['plain'][F.reason] == 'looks real'
    # The VLM verdict was obtained for both regardless of the write outcome
    # (this count reflects labeler calls, not writes -- see the router's
    # docstring/response contract).
    assert resp['verified'] == 2


@pytest.mark.asyncio
async def test_verify_regions_writes_normally_when_no_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.routers.curation.vlm as vlm_mod
    from src.routers.curation.vlm import VlmVerifyRegionsRequest

    fake = QueryFakeOpenSearch({ITEMS: {'plain': _item('plain')}})
    _setup(monkeypatch, _Labeler())

    await vlm_mod.vlm_verify_regions(VlmVerifyRegionsRequest(crop_ids=['plain']), fake)

    assert fake.bulk_calls == 1
    docs = fake.docs(ITEMS)
    assert docs['plain'][F.verified] is True
    assert docs['plain'][F.reason] == 'looks real'
