"""Tests for scripts/curation/backfill_region_embeddings.py.

No real Triton/OpenSearch/filesystem access — AsyncTritonPool, PEEncoder
and the disk-crop helper are all monkeypatched to fakes local to this
file, mirroring the pattern in test_revert_class_cluster_promotions.py.
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


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / 'scripts' / 'curation'
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import backfill_region_embeddings as backfill_script  # noqa: E402


class _FakeTritonPool:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None: ...

    async def initialize(self) -> None:
        return None


class _FakePE:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.embed_crops_calls: list[int] = []

    async def embed_crops(self, crops: list[Any], max_batch: int = 32) -> np.ndarray:  # noqa: ARG002
        self.embed_crops_calls.append(len(crops))
        return np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (len(crops), 1))


class _FakeOSClient:
    """Fake AsyncOpenSearch covering scroll + bulk for the backfill script."""

    def __init__(self, hits: list[dict[str, Any]]) -> None:
        self._hits = hits
        self.bulk_calls: list[dict[str, Any]] = []
        self.refreshed = False

    async def search(self, *, index: str, body: dict[str, Any], **kw: Any) -> dict[str, Any]:  # noqa: ARG002
        return {'_scroll_id': 'scroll-1', 'hits': {'hits': self._hits}}

    async def scroll(self, *, scroll_id: str, **kw: Any) -> dict[str, Any]:  # noqa: ARG002
        return {'_scroll_id': scroll_id, 'hits': {'hits': []}}

    async def clear_scroll(self, *, scroll_id: str, **kw: Any) -> dict[str, Any]:  # noqa: ARG002
        return {}

    async def bulk(self, *, body: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:  # noqa: ARG002
        for action, doc in zip(body[0::2], body[1::2], strict=True):
            self.bulk_calls.append({'id': action['update']['_id'], 'doc': doc['doc']})
        return {'errors': False, 'items': []}

    async def close(self) -> None:
        return None

    class indices:  # noqa: N801
        @staticmethod
        async def refresh(*, index: str) -> None:  # noqa: ARG004
            return None


def _tiny_jpeg() -> bytes:
    img = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), mode='RGB')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()


def _hit(doc_id: str, *, image_path: str | None, bbox: list[float] | None) -> dict[str, Any]:
    source: dict[str, Any] = {}
    if image_path is not None:
        source['image_path'] = image_path
    if bbox is not None:
        source['region_bbox_norm'] = bbox
    return {'_id': doc_id, '_source': source}


@pytest.mark.asyncio
async def test_dry_run_reports_count_and_does_not_touch_triton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hits = [_hit('crop-1', image_path='/data/a.jpg', bbox=[0.1, 0.1, 0.5, 0.5])]
    client = _FakeOSClient(hits)
    monkeypatch.setattr(backfill_script, 'AsyncOpenSearch', MagicMock(return_value=client))
    triton_pool_ctor = MagicMock(
        side_effect=AssertionError('Triton must not be touched in dry-run')
    )
    monkeypatch.setattr(backfill_script, 'AsyncTritonPool', triton_pool_ctor)

    rc = await backfill_script._run(
        'http://fake:9200', 'fake-triton:8001', apply=False, max_docs=None
    )

    assert rc == 0
    assert client.bulk_calls == []
    triton_pool_ctor.assert_not_called()


@pytest.mark.asyncio
async def test_apply_writes_normalized_embeddings_for_eligible_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hits = [
        _hit('crop-1', image_path='/data/a.jpg', bbox=[0.1, 0.1, 0.5, 0.5]),
        _hit('crop-2', image_path='/data/b.jpg', bbox=[0.2, 0.2, 0.6, 0.6]),
    ]
    client = _FakeOSClient(hits)
    fake_pe = _FakePE()
    monkeypatch.setattr(backfill_script, 'AsyncOpenSearch', MagicMock(return_value=client))
    monkeypatch.setattr(backfill_script, 'AsyncTritonPool', _FakeTritonPool)
    monkeypatch.setattr(backfill_script, 'PEEncoder', lambda **_kw: fake_pe)
    monkeypatch.setattr(
        backfill_script,
        '_crop_jpeg_from_disk',
        lambda image_path, bbox: _tiny_jpeg(),  # noqa: ARG005
    )

    rc = await backfill_script._run(
        'http://fake:9200', 'fake-triton:8001', apply=True, max_docs=None
    )

    assert rc == 0
    assert len(client.bulk_calls) == 2
    written_ids = {c['id'] for c in client.bulk_calls}
    assert written_ids == {'crop-1', 'crop-2'}
    for call in client.bulk_calls:
        vec = call['doc']['region_embedding']
        assert len(vec) == 3
        assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.asyncio
async def test_apply_skips_items_with_unreadable_source_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hits = [_hit('crop-missing', image_path='/data/gone.jpg', bbox=[0.1, 0.1, 0.5, 0.5])]
    client = _FakeOSClient(hits)
    fake_pe = _FakePE()
    monkeypatch.setattr(backfill_script, 'AsyncOpenSearch', MagicMock(return_value=client))
    monkeypatch.setattr(backfill_script, 'AsyncTritonPool', _FakeTritonPool)
    monkeypatch.setattr(backfill_script, 'PEEncoder', lambda **_kw: fake_pe)
    monkeypatch.setattr(
        backfill_script,
        '_crop_jpeg_from_disk',
        lambda image_path, bbox: None,  # noqa: ARG005
    )

    rc = await backfill_script._run(
        'http://fake:9200', 'fake-triton:8001', apply=True, max_docs=None
    )

    assert rc == 0
    assert client.bulk_calls == []
    assert fake_pe.embed_crops_calls == []


@pytest.mark.asyncio
async def test_selection_query_excludes_items_that_already_have_the_field() -> None:
    """Resumability: the query itself must exclude already-embedded items."""
    query = backfill_script._selection_query()
    assert {'exists': {'field': 'region_bbox_norm'}} in query['bool']['must']
    assert {'exists': {'field': 'region_embedding'}} in query['bool']['must_not']
