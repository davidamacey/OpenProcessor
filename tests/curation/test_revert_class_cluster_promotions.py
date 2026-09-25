"""Data repair: scripts/curation/revert_class_cluster_promotions.py.

Reproduces the exact bug shape: an item whose class_source was set to
``cluster_majority_agreement`` by ``auto_promote`` on a class-range
cluster_id (cluster_id == class_id, purity trivially 1.0) must be
reverted to its prior class assignment; an item promoted from a real
candidate cluster (cluster_id >= RESIDUAL_CLUSTER_ID_OFFSET) is left
alone -- that promotion was legitimate.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from curation.occ_fakes import make_bulk_response, make_bulk_update_item, make_mget_response


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / 'scripts' / 'curation'
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import revert_class_cluster_promotions as revert_script  # noqa: E402


def _history_entry(**overrides: Any) -> dict[str, Any]:
    base = {
        'class_id': 3,
        'class_name': 'class_b',
        'class_source': 'classifier_model',
        'label_source': 'classifier_model',
        'confidence': 0.62,
        'writer': 'auto_promote',
        'at': '2026-09-01T00:00:00+00:00',
    }
    base.update(overrides)
    return base


class TestLastAutoPromoteEntry:
    def test_matches_class_range_auto_promote(self) -> None:
        source = {
            'cluster_id': 3,  # class-range: cluster_id == class_id
            'class_id_history': [_history_entry()],
        }
        entry = revert_script._last_auto_promote_entry(source)
        assert entry is not None
        assert entry['class_name'] == 'class_b'

    def test_ignores_candidate_range_promotions(self) -> None:
        """A promotion out of a real candidate cluster (>= the residual
        offset) is legitimate and must not revert."""
        source = {
            'cluster_id': revert_script.RESIDUAL_CLUSTER_ID_OFFSET + 5,
            'class_id_history': [_history_entry()],
        }
        assert revert_script._last_auto_promote_entry(source) is None

    def test_ignores_non_auto_promote_writer(self) -> None:
        source = {
            'cluster_id': 3,
            'class_id_history': [_history_entry(writer='human:label_crop')],
        }
        assert revert_script._last_auto_promote_entry(source) is None

    def test_ignores_empty_history(self) -> None:
        assert revert_script._last_auto_promote_entry({'cluster_id': 3}) is None


class _FakeRevertClient:
    """Fake AsyncOpenSearch covering scroll (ids+source) + OCC-bulk write."""

    def __init__(self, hits: list[dict[str, Any]]) -> None:
        self._hits = hits
        self.bulk_calls: list[dict[str, Any]] = []

    async def search(self, *, index: str, body: dict[str, Any], **kw: Any) -> dict[str, Any]:  # noqa: ARG002
        return {'_scroll_id': 'scroll-1', 'hits': {'hits': self._hits}}

    async def scroll(self, *, scroll_id: str, **kw: Any) -> dict[str, Any]:  # noqa: ARG002
        return {'_scroll_id': scroll_id, 'hits': {'hits': []}}

    async def clear_scroll(self, *, scroll_id: str, **kw: Any) -> dict[str, Any]:  # noqa: ARG002
        return {}

    async def mget(self, *, body: dict[str, Any]) -> dict[str, Any]:
        by_id = {h['_id']: h['_source'] for h in self._hits}
        found = {d['_id']: by_id[d['_id']] for d in body['docs'] if d['_id'] in by_id}
        return make_mget_response(found)

    async def bulk(self, *, body: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:  # noqa: ARG002
        items = []
        for action, doc in zip(body[0::2], body[1::2], strict=True):
            doc_id = action['update']['_id']
            self.bulk_calls.append({'id': doc_id, 'doc': doc['doc']})
            items.append(make_bulk_update_item(doc_id, status=200))
        return make_bulk_response(items)

    async def close(self) -> None:
        return None


def _hit(doc_id: str, source: dict[str, Any]) -> dict[str, Any]:
    return {'_id': doc_id, '_source': source}


@pytest.mark.asyncio
async def test_dry_run_does_not_write(monkeypatch: pytest.MonkeyPatch) -> None:
    hits = [
        _hit(
            'crop-class-range',
            {
                'cluster_id': 3,
                'class_name': 'class_b',
                'class_source': 'cluster_majority_agreement',
                'class_id_history': [_history_entry()],
            },
        ),
    ]
    client = _FakeRevertClient(hits)
    monkeypatch.setattr(revert_script, 'AsyncOpenSearch', MagicMock(return_value=client))

    rc = await revert_script._run('http://fake:9200', apply=False)

    assert rc == 0
    assert client.bulk_calls == []


@pytest.mark.asyncio
async def test_apply_reverts_only_class_range_promotions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hits = [
        _hit(
            'crop-class-range',
            {
                'cluster_id': 3,
                'class_name': 'class_b',
                'class_source': 'cluster_majority_agreement',
                'class_id_history': [_history_entry()],
            },
        ),
        _hit(
            'crop-candidate-range',
            {
                'cluster_id': revert_script.RESIDUAL_CLUSTER_ID_OFFSET + 1,
                'class_name': 'sportycar',
                'class_source': 'cluster_majority_agreement',
                'class_id_history': [
                    _history_entry(class_id=9, class_name='sportycar', confidence=0.7)
                ],
            },
        ),
    ]
    client = _FakeRevertClient(hits)
    monkeypatch.setattr(revert_script, 'AsyncOpenSearch', MagicMock(return_value=client))

    rc = await revert_script._run('http://fake:9200', apply=True)

    assert rc == 0
    assert len(client.bulk_calls) == 1
    written = client.bulk_calls[0]
    assert written['id'] == 'crop-class-range'
    assert written['doc']['class_id'] == 3
    assert written['doc']['class_name'] == 'class_b'
    assert written['doc']['class_source'] == 'classifier_model'
    assert written['doc']['class_validated'] is False
