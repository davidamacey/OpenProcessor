"""Tests for scripts/curation/backfill_region_embeddings.py (per-box vectors).

The OpenSearch fake is the shared query-semantics one (nested box queries,
mget, conditional bulk); Triton, the PE encoder and the disk-crop helper
are monkeypatched to fakes local to this file.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from curation.query_fakes import QueryFakeOpenSearch, matches
from src.config import get_region_fields
from src.services.curation.region_box_embeddings import current_vectors, entry_for
from src.services.curation.region_boxes import RegionBox, boxes_write_fields


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / 'scripts' / 'curation'
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import backfill_region_embeddings as backfill_script  # noqa: E402


F = get_region_fields()
INDEX = 'items'


class _OS(QueryFakeOpenSearch):
    async def close(self) -> None:
        return None

    class indices:  # noqa: N801
        @staticmethod
        async def refresh(*, index: str) -> None:  # noqa: ARG004
            return None


class _FakeTritonPool:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None: ...

    async def initialize(self) -> None:
        return None


class _FakePE:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.embed_crops_calls: list[int] = []

    async def embed_crops(self, crops: list[Any], max_batch: int = 32) -> np.ndarray:  # noqa: ARG002
        self.embed_crops_calls.append(len(crops))
        return np.tile(np.array([0.6, 0.8, 0.0], dtype=np.float32), (len(crops), 1))


def _tiny_jpeg() -> bytes:
    img = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), mode='RGB')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()


def _box(box_id: str, state: str = 'accepted', x: float = 0.1) -> RegionBox:
    return RegionBox(box_id=box_id, bbox_norm=(x, 0.1, x + 0.2, 0.4), state=state)


def _item(crop_id: str, boxes: list[RegionBox], **extra: Any) -> dict[str, Any]:
    return {
        'crop_id': crop_id,
        'image_path': f'/data/{crop_id}.jpg',
        **boxes_write_fields(boxes, current_src={}),
        **extra,
    }


def _patch(monkeypatch: pytest.MonkeyPatch, client: _OS, *, jpeg: bytes | None) -> _FakePE:
    fake_pe = _FakePE()
    monkeypatch.setattr(backfill_script, 'make_script_opensearch', MagicMock(return_value=client))
    monkeypatch.setattr(backfill_script, 'AsyncTritonPool', _FakeTritonPool)
    monkeypatch.setattr(backfill_script, 'PEEncoder', lambda **_kw: fake_pe)
    monkeypatch.setattr(backfill_script, 'get_curation_config', lambda: _Cfg())
    monkeypatch.setattr(
        backfill_script,
        '_crop_jpeg_from_disk',
        lambda image_path, bbox: jpeg,  # noqa: ARG005
    )
    return fake_pe


class _Cfg:
    items_index = INDEX


async def _run(*, apply: bool) -> int:
    return await backfill_script._run('http://fake:9200', 'fake:8001', apply=apply, max_docs=None)


@pytest.mark.asyncio
async def test_dry_run_does_not_touch_triton_or_write(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _OS({INDEX: {'c1': _item('c1', [_box('b1')])}})
    _patch(monkeypatch, client, jpeg=_tiny_jpeg())
    ctor = MagicMock(side_effect=AssertionError('Triton must not be touched in dry-run'))
    monkeypatch.setattr(backfill_script, 'AsyncTritonPool', ctor)

    assert await _run(apply=False) == 0

    assert client.bulk_calls == 0
    assert F.box_embeddings not in client.docs(INDEX)['c1']


@pytest.mark.asyncio
async def test_apply_embeds_every_missing_box_not_one_per_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boxes = [_box('b1'), _box('b2', x=0.5), _box('b3', 'false_positive', x=0.7)]
    client = _OS({INDEX: {'c1': _item('c1', boxes)}})
    fake_pe = _patch(monkeypatch, client, jpeg=_tiny_jpeg())

    assert await _run(apply=True) == 0

    doc = client.docs(INDEX)['c1']
    vectors = current_vectors(doc)
    assert set(vectors) == {'b1', 'b2', 'b3'}
    assert fake_pe.embed_crops_calls == [3]
    assert all(vec == pytest.approx([0.6, 0.8, 0.0]) for vec in vectors.values())
    # The box list and revision are never written by the backfill.
    assert doc[F.boxes] == _item('c1', boxes)[F.boxes]
    assert doc[F.revision] == _item('c1', boxes)[F.revision]


@pytest.mark.asyncio
async def test_apply_is_resumable_and_re_embeds_a_moved_box(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    b1, b2 = _box('b1'), _box('b2', x=0.5)
    moved_b2 = _box('b2', x=0.6)
    stored = [entry_for(b1, [1.0, 0.0, 0.0]), entry_for(b2, [0.0, 1.0, 0.0])]
    client = _OS({INDEX: {'c1': _item('c1', [b1, moved_b2], **{F.box_embeddings: stored})}})
    fake_pe = _patch(monkeypatch, client, jpeg=_tiny_jpeg())

    assert await _run(apply=True) == 0

    vectors = current_vectors(client.docs(INDEX)['c1'])
    assert fake_pe.embed_crops_calls == [1]
    assert vectors['b1'] == [1.0, 0.0, 0.0]
    assert vectors['b2'] != [0.0, 1.0, 0.0]


@pytest.mark.asyncio
async def test_apply_skips_boxes_whose_source_image_is_unreadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _OS({INDEX: {'c1': _item('c1', [_box('b1')])}})
    fake_pe = _patch(monkeypatch, client, jpeg=None)

    assert await _run(apply=True) == 0

    assert client.bulk_calls == 0
    assert fake_pe.embed_crops_calls == []


def test_selection_query_matches_only_items_with_an_embeddable_box() -> None:
    """Rejected / proposed boxes carry no vector, so an item holding only
    those is never selected. Asserted against the real nested-match
    semantics, with the *same* box satisfying the state filter."""
    docs = {
        'fp-only': _item('fp-only', [_box('b1', 'false_positive')]),
        'accepted': _item('accepted', [_box('b1')]),
        'rejected-only': _item('rejected-only', [_box('b1', 'rejected')]),
        'proposed-only': _item('proposed-only', [_box('b1', 'proposed')]),
    }
    query = backfill_script._selection_query()
    assert {k for k, d in docs.items() if matches(d, query)} == {'fp-only', 'accepted'}
