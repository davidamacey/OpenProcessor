"""A drain never spends model work on an item at the nested-document limit (OpenSearch k-NN issue
3623): it refuses every machine write, so the item would be picked up on every cycle for ever.
The pending fetch drops it before any task exists, remembers it, keeps it out of the next query and
the cycle ends idle."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

from scripts.curation import vlm_pending
from scripts.curation.worker.cascade import _fetch_pending
from src.config import get_region_fields
from src.services.curation import over_limit_memo
from src.services.curation.over_limit_memo import OverLimitMemo


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

F = get_region_fields()


class _Search:
    """Answers each search with the next page of hits; records the bodies."""

    def __init__(self, *pages: list[dict[str, Any]]) -> None:
        self.pages = list(pages)
        self.bodies: list[dict[str, Any]] = []

    async def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self.bodies.append(body)
        page = self.pages.pop(0) if self.pages else []
        return {'hits': {'hits': page}}


def _hit(crop_id: str, n_boxes: int) -> dict[str, Any]:
    return {
        '_id': crop_id,
        '_source': {
            'crop_id': crop_id,
            'image_path': '/x.jpg',
            'bbox_norm': [0.1, 0.1, 0.5, 0.5],
            F.status: 'pending_detection',
            F.boxes: [
                {'box_id': f'b{i}', 'bbox_norm': [0.1, 0.1, 0.2, 0.2], 'state': 'accepted'}
                for i in range(n_boxes)
            ],
        },
    }


@pytest.fixture(autouse=True)
def memo(monkeypatch: pytest.MonkeyPatch) -> OverLimitMemo:
    memo = OverLimitMemo()
    monkeypatch.setattr(over_limit_memo, 'MEMO', memo)
    monkeypatch.setattr('scripts.curation.worker.cascade.MEMO', memo)
    monkeypatch.setattr(vlm_pending, 'MEMO', memo)
    return memo


@pytest.mark.asyncio
async def test_a_pending_item_over_the_limit_makes_no_task_and_is_counted(
    memo: OverLimitMemo,
) -> None:
    client = _Search([_hit('big', 80)])
    assert await _fetch_pending(cast('AsyncOpenSearch', client), batch_size=10) == []
    assert memo.skipped == 1


@pytest.mark.asyncio
async def test_the_next_cycle_keeps_the_item_out_of_the_query() -> None:
    client = _Search([_hit('big', 80)], [])
    await _fetch_pending(cast('AsyncOpenSearch', client), batch_size=10)
    await _fetch_pending(cast('AsyncOpenSearch', client), batch_size=10)
    assert {'ids': {'values': ['big']}} in client.bodies[1]['query']['bool']['must_not']


@pytest.mark.asyncio
async def test_a_full_page_of_over_limit_items_does_not_hide_the_work_behind_it() -> None:
    client = _Search([_hit('big1', 80), _hit('big2', 90)], [_hit('ok', 3)])
    tasks = await _fetch_pending(cast('AsyncOpenSearch', client), batch_size=2)
    assert [t.crop_id for t in tasks] == ['ok']
    ids = client.bodies[1]['query']['bool']['must_not'][0]['ids']['values']
    assert sorted(ids) == ['big1', 'big2']


@pytest.mark.asyncio
async def test_an_item_just_below_the_limit_is_worked() -> None:
    client = _Search([_hit('edge', 74)])
    assert [
        t.crop_id for t in await _fetch_pending(cast('AsyncOpenSearch', client), batch_size=10)
    ] == ['edge']


def test_the_memo_forgets_an_item_after_its_ttl() -> None:
    now = [0.0]
    memo = OverLimitMemo(ttl_s=10, clock=lambda: now[0])
    kept, over = memo.split([_hit('big', 80)], ['region_box_embeddings'])
    assert (kept, over) == ([], ['big'])
    assert memo.excluded() == ['big']
    now[0] = 11.0
    assert memo.excluded() == []


@pytest.mark.asyncio
async def test_the_vlm_worker_fetch_drops_an_over_limit_item_before_any_model_call() -> None:
    from src.services.curation.vlm_scope import ScopeCache

    class _Scope(ScopeCache):
        async def get(self, _os: Any, _slug: str) -> Any:
            from src.services.curation.vlm_policy import VlmPolicy

            return VlmPolicy(), None

    client = _Search([_hit('big', 80), _hit('ok', 3)])
    ids = await vlm_pending.fetch_pending_ids(
        client,
        slug='default',
        scope_cache=_Scope(),
        batch_size=2,
        classifier_skip_conf=0.8,
    )
    assert ids == ['ok']
