"""An issue's ``bypassable`` flag must match the gate that validates it.

Each validator owns its ``BYPASSABLE_CODES`` (the codes ``force: true`` gets
past on activation) but builds issues with the shared ``_issue`` helper. The
per-issue flag drives the GUI's "(can be overridden)" hint, so it must come
from the same set as ``force_allowed``, never from another validator's set.
"""

from __future__ import annotations

import pytest

from curation._ingest_profile_world import Pool, body


@pytest.mark.asyncio
async def test_ingest_profile_not_ready_model_is_not_marked_bypassable() -> None:
    from src.services.config_store.ingest_profile_validation import validate_ingest_profile

    pool = Pool()
    pool.ready['birds'] = False
    report = await validate_ingest_profile(
        'coco-small', body('birds'), for_activation=True, pool_provider=lambda: pool
    )

    assert [e.code for e in report.errors] == ['detector_model_not_ready']
    assert report.force_allowed is False
    assert report.errors[0].bypassable is False


@pytest.mark.asyncio
async def test_ingest_profile_triton_outage_is_marked_bypassable() -> None:
    from src.services.config_store.ingest_profile_validation import validate_ingest_profile

    class DownPool(Pool):
        async def is_model_ready(self, _name: str) -> bool:
            raise OSError('connection refused')

    pool = DownPool()
    report = await validate_ingest_profile(
        'coco-small', body('birds'), for_activation=True, pool_provider=lambda: pool
    )

    assert [e.code for e in report.errors] == ['triton_unreachable']
    assert report.force_allowed is True
    assert report.errors[0].bypassable is True


@pytest.mark.asyncio
async def test_open_vocab_unconfigured_segmenter_is_marked_bypassable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.services.config_store import open_vocab_validation

    monkeypatch.delenv('OP_SEGMENTER_URL', raising=False)
    issues = await open_vocab_validation._check_segmenter(True, None)

    assert [i.code for i in issues] == ['segmenter_not_configured']
    assert 'segmenter_not_configured' in open_vocab_validation.BYPASSABLE_CODES
    assert issues[0].bypassable is True
