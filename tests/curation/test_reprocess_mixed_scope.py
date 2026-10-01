"""``detect`` over more images than run inline is a background job; region
and vlm would flip ahead of it, so that mix is refused rather than reordered."""

from __future__ import annotations

import pytest

from curation.reprocess_fixtures import make_fake
from src.services.curation.reprocess import plan_reprocess
from src.services.curation.reprocess_models import ReprocessRequest, ReprocessTargets
from src.services.curation.reprocess_targets import ReprocessTargetsError


def _request(scopes: list[str], images: int) -> ReprocessRequest:
    return ReprocessRequest(
        targets=ReprocessTargets(image_ids=[f'img-{i}' for i in range(images)]),
        scopes=scopes,  # type: ignore[arg-type]
        dry_run=True,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('other', ['region', 'vlm'])
async def test_detect_with_region_or_vlm_above_the_sync_limit_is_refused(
    monkeypatch: pytest.MonkeyPatch, other: str
) -> None:
    monkeypatch.setenv('OP_REPROCESS_SYNC_MAX', '2')
    with pytest.raises(ReprocessTargetsError, match='background job'):
        await plan_reprocess(make_fake([]), _request(['detect', other], 3))


@pytest.mark.asyncio
async def test_the_same_mix_within_the_sync_limit_runs_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('OP_REPROCESS_SYNC_MAX', '2')
    plan = await plan_reprocess(make_fake([]), _request(['detect', 'region'], 2))
    assert [r.scope for r in plan.results] == ['detect', 'region']


@pytest.mark.asyncio
async def test_a_large_detect_alone_or_with_embed_is_still_one_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('OP_REPROCESS_SYNC_MAX', '2')
    plan = await plan_reprocess(make_fake([]), _request(['detect', 'embed'], 3))
    assert [r.scope for r in plan.results] == ['detect', 'embed']
