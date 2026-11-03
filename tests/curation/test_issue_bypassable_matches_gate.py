"""An issue's ``bypassable`` flag must match the gate that validates it.

Each validator owns its ``BYPASSABLE_CODES`` (the codes ``force: true`` gets
past on activation) but builds issues with ``issue_builder(BYPASSABLE_CODES)``, a required set. The
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


@pytest.mark.asyncio
async def test_shared_prompt_check_in_open_vocab_uses_the_validators_own_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``check_segmenter_prompt_text`` is shared with the region validator, so it must stamp
    with the calling validator's set. Widen the open-vocab set by one code the region set
    lacks: the issue the shared helper raises must follow it."""
    from src.services.config_store import open_vocab_validation
    from src.services.config_store.profile_issues import (
        BYPASSABLE_CODES as REGION_CODES,
        issue_builder,
    )

    assert 'segmenter_prompt_too_long' not in REGION_CODES
    widened = open_vocab_validation.BYPASSABLE_CODES | {'segmenter_prompt_too_long'}
    monkeypatch.setattr(open_vocab_validation, '_issue', issue_builder(widened))

    report = await open_vocab_validation.validate_open_vocab(
        'street', {'targets': [{'prompt': 'x' * 201, 'class_name': 'cone'}]}
    )

    too_long = [e for e in report.errors if e.code == 'segmenter_prompt_too_long']
    assert too_long
    assert too_long[0].bypassable is True


def test_shared_prompt_check_follows_the_builder_it_is_given() -> None:
    from src.services.config_store.profile_dependency_checks import check_segmenter_prompt_text
    from src.services.config_store.profile_issues import issue_builder

    never = issue_builder(frozenset())
    always = issue_builder(frozenset({'segmenter_prompt_empty'}))

    assert not check_segmenter_prompt_text('', sole_leg=True, issue=never)[0].bypassable
    assert check_segmenter_prompt_text('', sole_leg=True, issue=always)[0].bypassable
