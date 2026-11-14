"""The cluster projection scripts do not rewrite an item at the nested-document limit (OpenSearch
k-NN issue 3623), and the job counts those items apart from real failures."""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from src.services.clustering.projection import _bulk, update_action


ROW = {'uid': None, 'state': 'unassigned', 'score': None, 'margin': None}


@pytest.mark.parametrize('target', ['item', 'region_box'])
def test_both_scripts_carry_the_limit_guard(target: str) -> None:
    key = 'crop1:b1' if target == 'region_box' else 'crop1'
    _action, body = update_action('items', target, key, uuid.uuid4(), ROW)
    source = body['script']['source']
    assert 'op_nested >= 150' in source
    assert "throw new IllegalStateException('op_nested_limit')" in source
    assert 'params.pid' in source  # the original fence is still inside the guard


class _Client:
    def __init__(self, items: list[dict[str, Any]]) -> None:
        self._items = items

    async def bulk(self, *, body: list[dict[str, Any]], refresh: bool = False) -> dict[str, Any]:
        return {'errors': True, 'items': self._items}


@pytest.mark.asyncio
async def test_bulk_counts_over_limit_items_apart_from_failures() -> None:
    over = {
        'update': {
            '_id': 'a',
            'error': {
                'type': 'script_exception',
                'caused_by': {'type': 'illegal_state_exception', 'reason': 'op_nested_limit'},
            },
        }
    }
    broken = {'update': {'_id': 'b', 'error': {'type': 'boom'}}}
    fine = {'update': {'_id': 'c', 'result': 'updated'}}
    client = _Client([over, broken, fine])
    action, body = update_action('items', 'item', 'a', uuid.uuid4(), ROW)
    assert await _bulk(client, [(action, body)]) == (1, 1)  # type: ignore[arg-type]
