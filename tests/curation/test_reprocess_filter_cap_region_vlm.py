"""``limit`` / ``sample`` on a filter target cap the region and vlm scopes
too, not only the image scopes: a sample of 2 re-queues or clears exactly 2."""

from __future__ import annotations

from typing import Any

import pytest

from curation.reprocess_fixtures import F, RegionStatus, box, docs, item, make_fake
from src.services.curation.reprocess import apply_reprocess
from src.services.curation.reprocess_models import (
    ReprocessFilter,
    ReprocessRequest,
    ReprocessTargets,
)


FAILED = RegionStatus.DETECTION_FAILED.value
VLM_HISTORY = [
    {
        'class_id': None,
        'class_name': None,
        'class_source': 'unlabeled_proposal',
        'label_source': 'unlabeled_proposal',
        'class_validated': False,
        'cluster_id': None,
        'writer': 'vlm_label_batch',
        'restorable': True,
        'at': 't0',
    }
]


def _request(scope: str, flt: ReprocessFilter, *, dry_run: bool) -> ReprocessRequest:
    return ReprocessRequest(
        targets=ReprocessTargets(filter=flt, limit=2, sample='random', seed=1),
        scopes=[scope],  # type: ignore[list-item]
        dry_run=dry_run,
    )


def _region_items() -> list[dict[str, Any]]:
    return [item(f'r{i}', FAILED, boxes=(box(f'b{i}'),)) for i in range(10)]


def _vlm_items() -> list[dict[str, Any]]:
    return [
        item(f'v{i}', class_source='vlm', class_id=1, class_name='widget', history=VLM_HISTORY)
        for i in range(10)
    ]


@pytest.mark.asyncio
async def test_region_scope_requeues_exactly_the_sampled_ids() -> None:
    flt = ReprocessFilter(region_status=[FAILED])
    dry = await apply_reprocess(make_fake(_region_items()), _request('region', flt, dry_run=True))
    assert dry.scopes[0].selected == 2

    fake = make_fake(_region_items())
    resp = await apply_reprocess(fake, _request('region', flt, dry_run=False))
    moved = {
        cid for cid, d in docs(fake).items() if d[F.status] == RegionStatus.PENDING_DETECTION.value
    }
    assert resp.scopes[0].queued == 2
    assert len(moved) == 2


@pytest.mark.asyncio
async def test_vlm_scope_clears_exactly_the_sampled_ids() -> None:
    flt = ReprocessFilter(class_names=['widget'])
    dry = await apply_reprocess(make_fake(_vlm_items()), _request('vlm', flt, dry_run=True))
    assert dry.scopes[0].selected == 2

    fake = make_fake(_vlm_items())
    resp = await apply_reprocess(fake, _request('vlm', flt, dry_run=False))
    cleared = {cid for cid, d in docs(fake).items() if d['class_source'] != 'vlm'}
    assert resp.scopes[0].queued == 2
    assert len(cleared) == 2


@pytest.mark.asyncio
async def test_region_and_dry_run_pick_the_same_seeded_ids() -> None:
    flt = ReprocessFilter(region_status=[FAILED])
    runs = []
    for _ in range(2):
        fake = make_fake(_region_items())
        await apply_reprocess(fake, _request('region', flt, dry_run=False))
        runs.append(
            {
                cid
                for cid, d in docs(fake).items()
                if d[F.status] == RegionStatus.PENDING_DETECTION.value
            }
        )
    assert runs[0] == runs[1]
