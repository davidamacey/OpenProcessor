"""The auto-label embed stage initializes the pool it builds, and surfaces total failure."""

from __future__ import annotations

from typing import Any

import pytest

from src.services.curation.autolabel import embed_stage


class _Plan:
    image_ids = ['a', 'b']
    embed = object()


class _StrictPool:
    """Raises like the real pool when used before ``initialize``."""

    instances: list[_StrictPool] = []

    def __init__(self, **_kw: Any) -> None:
        self.ready = False
        self.closed = False
        _StrictPool.instances.append(self)

    async def initialize(self) -> None:
        self.ready = True

    async def close(self) -> None:
        self.closed = True


class _PoolEncoder:
    def __init__(self, triton_pool: _StrictPool) -> None:
        self.pool = triton_pool


async def _plan(*_a: Any, **_k: Any) -> _Plan:
    return _Plan()


async def _docs(*_a: Any, **_k: Any) -> list[dict[str, Any]]:
    return [{}]


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch) -> None:
    _StrictPool.instances.clear()
    monkeypatch.setattr('src.clients.triton_pool.AsyncTritonPool', _StrictPool)
    monkeypatch.setattr('src.clients.pe_encoder.PEEncoder', _PoolEncoder)
    monkeypatch.setattr(embed_stage, 'plan_reprocess', _plan)
    monkeypatch.setattr(embed_stage, 'existing_images', _docs)
    monkeypatch.setattr(embed_stage, '_images_index', lambda: 'images')


@pytest.mark.asyncio
async def test_owned_pool_is_initialized_before_use(
    patched: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _embed(_os: Any, encoder: _PoolEncoder, _docs: Any, _opts: Any, result: Any) -> None:
        if not encoder.pool.ready:
            raise RuntimeError('AsyncTritonPool not initialized. Call initialize() first.')
        result.queued += 1

    monkeypatch.setattr(embed_stage, 'embed_image_chunk', _embed)
    out = await embed_stage.run_embed_missing_stage(
        None, class_id=None, cluster_id=None, item_filter=None
    )
    assert out['images'] == 1
    assert 'status' not in out
    assert _StrictPool.instances[0].closed


@pytest.mark.asyncio
async def test_every_image_failing_is_an_error_stage(
    patched: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _embed(_os: Any, _enc: Any, _docs: Any, _opts: Any, result: Any) -> None:
        result.failed += 2

    monkeypatch.setattr(embed_stage, 'embed_image_chunk', _embed)
    out = await embed_stage.run_embed_missing_stage(
        None, class_id=None, cluster_id=None, item_filter=None
    )
    assert out['status'] == 'error'
    assert out['images_failed'] == 2
