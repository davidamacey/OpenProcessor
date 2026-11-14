"""``find_items_over_nested_limit.py``: lists the items at or over the nested-document limit
(OpenSearch k-NN issue 3623), exits 1 when any exist, 0 when none, and never writes."""

from __future__ import annotations

import argparse
from typing import Any

import pytest

import scripts.curation.find_items_over_nested_limit as script
from src.config import get_region_fields
from src.services.curation.nested_limit_scan import find_over_limit_items


F = get_region_fields()
EXCLUDES = ['pe_embedding', F.box_embeddings]


def _item(n_boxes: int, **extra: Any) -> dict[str, Any]:
    return {F.boxes: [{'box_id': f'b{i}'} for i in range(n_boxes)], **extra}


class _Client:
    """Pages hits by ``search_after`` over a fixed id-ordered store; records every call."""

    def __init__(self, items: dict[str, dict[str, Any]]) -> None:
        self.items = dict(sorted(items.items()))
        self.calls: list[str] = []

    async def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self.calls.append('search')
        ids = list(self.items)
        after = body.get('search_after')
        start = ids.index(after[0]) + 1 if after else 0
        hits = [
            {'_id': i, '_source': self.items[i], 'sort': [i]}
            for i in ids[start : start + body['size']]
        ]
        return {'hits': {'hits': hits}}

    async def close(self) -> None:
        self.calls.append('close')


@pytest.mark.asyncio
async def test_lists_every_item_at_or_over_the_limit_with_its_numbers() -> None:
    client = _Client({'a': _item(74), 'b': _item(75), 'c': _item(200), 'd': _item(1)})
    found = await find_over_limit_items(
        client, index='items', vector_excludes=EXCLUDES, page_size=2
    )
    assert [(f.crop_id, f.boxes, f.nested_docs, f.remove_at_least) for f in found] == [
        ('b', 75, 150, 1),
        ('c', 200, 400, 126),
    ]


@pytest.mark.asyncio
async def test_assist_attributes_count_and_lower_the_boxes_an_item_may_keep() -> None:
    attrs = {'attributes': [{'key': 'k', 'value': 'v'}] * 10}
    client = _Client({'a': _item(70, assist=attrs)})  # 140 + 10
    [found] = await find_over_limit_items(client, index='items', vector_excludes=EXCLUDES)
    assert (found.nested_docs, found.remove_at_least) == (150, 1)  # 69 boxes + 10 = 148


def _args(**kw: Any) -> argparse.Namespace:
    return argparse.Namespace(index='items', page_size=500, repair_plan=False, **kw)


@pytest.fixture(autouse=True)
def _excludes(monkeypatch: pytest.MonkeyPatch) -> None:
    async def excludes() -> list[str]:
        return EXCLUDES

    monkeypatch.setattr(script, 'vector_source_excludes', excludes)


@pytest.mark.asyncio
async def test_exit_code_is_1_when_an_item_is_over_the_limit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = _Client({'ok': _item(5), 'big': _item(90)})
    assert await script.run(_args(), client) == 1
    out = capsys.readouterr().out
    assert 'big  boxes=90  nested_docs=180' in out
    assert 'must be repaired before their box vectors are trusted' in out
    assert 'PUT /curation/projects/{project}/crops/{crop_id}/regions' in out
    assert 'ok' not in out.split('1 item(s)')[0]


@pytest.mark.asyncio
async def test_exit_code_is_0_when_nothing_is_over_the_limit() -> None:
    assert await script.run(_args(), _Client({'ok': _item(74)})) == 0


@pytest.mark.asyncio
async def test_repair_plan_prints_how_many_boxes_each_item_must_lose(
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = _args()
    args.repair_plan = True
    await script.run(args, _Client({'big': _item(90)}))
    assert 'remove_at_least=16' in capsys.readouterr().out


@pytest.mark.asyncio
async def test_the_scan_only_searches() -> None:
    client = _Client({'big': _item(90)})
    await script.run(_args(), client)
    assert set(client.calls) == {'search'}
