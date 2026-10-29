"""setp_inventory: per-project counts from the projects and stats routes, read only."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import httpx

from scripts.bench import setp_inventory as inv


if TYPE_CHECKING:
    from pathlib import Path

STATS = {
    'alpha': {'items': 900, 'images': 400, 'validated': 120, 'holdout_items': 40, 'classes': 7},
    'beta': {'items': 5, 'images': 5, 'validated': None, 'holdout_items': None, 'classes': 1},
}


def _handler(seen: list[tuple[str, str]]):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        if request.url.path == '/curation/projects':
            assert request.url.params['include_archived'] == 'true'
            return httpx.Response(
                200,
                json={
                    'projects': [
                        {'slug': 'alpha', 'status': 'active'},
                        {'slug': 'beta', 'status': 'archived'},
                    ]
                },
            )
        slug = request.url.path.split('/')[3]
        return httpx.Response(200, json={'counts': STATS[slug]})

    return handler


def test_inventory_lists_every_project_with_counts_only(tmp_path: Path) -> None:
    seen: list[tuple[str, str]] = []
    out = tmp_path / 'w0' / 'setp_inventory.json'
    rows = inv.collect('http://x', '/curation', httpx.MockTransport(_handler(seen)))
    assert {m for m, _ in seen} == {'GET'}  # read only
    assert [r['slug'] for r in rows] == ['alpha', 'beta']
    assert rows[0] == {
        'slug': 'alpha',
        'status': 'active',
        'items': 900,
        'images': 400,
        'validated': 120,
        'holdout_items': 40,
        'classes': 7,
    }
    assert rows[1]['validated'] is None  # unknown stays unknown, not zero
    inv.write(out, rows)
    assert json.loads(out.read_text())['projects'] == rows
