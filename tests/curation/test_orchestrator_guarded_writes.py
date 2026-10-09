"""The class-clustering orchestrator's bulk writers must never clobber a
concurrent human label/verification/exclusion.

Two writers fetch candidates, spend seconds-to-minutes fitting a model,
then bulk-write ``cluster_id``/``cluster_subid`` back. A blind
``{'doc': {...}}`` update would silently overwrite a human write that
landed during the fit, so both (``cluster_residuals``,
``assign_only_residuals``) share :func:`_guarded_class_cluster_write`, a
painless ``script`` update that noops when the doc's current state shows
human ownership; ``_bulk_update_subids`` (item refine) has its own "doc
moved to a different cluster" guard. The region-box clustering writers are
guarded in ``region_box_rows.write_box_edits`` (see
``test_region_box_clustering.py``).

The guard logic lives in a single ``GuardClause`` list
(``CLASS_CLUSTER_WRITE_GUARD_CLAUSES``), rendered into painless text by
``_guard_condition_painless`` and evaluated in plain Python by
``_guard_condition_matches`` -- both read off the *same* list, so there is
nothing for a "predicate vs. script text" consistency test to catch
drifting apart; the tests below instead prove the list matches
``is_locked_class`` and that the writers actually build ``script``
actions, not ``doc`` actions.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.clients.occ import is_locked_class
from src.services.curation.clustering import cluster_write_guard as guard, refine


# ---------------------------------------------------------------------------
# Class-cluster write guard (cluster_residuals / assign_only_residuals)
# ---------------------------------------------------------------------------


def test_guarded_class_cluster_write_builds_a_script_action_not_doc() -> None:
    action = guard._guarded_class_cluster_write(10042, 0.12)
    assert 'script' in action
    assert 'doc' not in action
    assert action['script']['lang'] == 'painless'
    assert action['script']['params'] == {'cid': 10042, 'dist': 0.12}


def test_guarded_class_cluster_write_script_sets_expected_fields() -> None:
    src = guard._guarded_class_cluster_write(10042, 0.12)['script']['source']
    assert "ctx._source['cluster_id'] = params.cid" in src
    assert "ctx._source.remove('cluster_subid')" in src
    assert "ctx._source['cluster_distance'] = params.dist" in src
    assert "ctx.op = 'noop'" in src


def test_class_cluster_write_guard_matches_is_locked_class() -> None:
    """Cross-check: every source is_locked_class flags as human-owned
    must also be flagged by CLASS_CLUSTER_WRITE_GUARD_CLAUSES (the list
    that renders into the painless guard), for a representative sample of
    the markers occ.py documents (human, human_move, vlm_human_confirmed)
    plus the class_validated / class_excluded guards the script adds on
    top."""
    samples: list[dict[str, Any]] = [
        {'class_source': 'human'},
        {'class_source': 'human_move'},
        {'class_source': 'vlm_human_confirmed'},
        {'class_validated': True, 'class_source': 'item_model'},
        {'class_excluded': True, 'class_source': 'item_model'},
    ]
    for source in samples:
        human_owned = is_locked_class(source) or bool(
            source.get('class_validated') or source.get('class_excluded')
        )
        assert human_owned, source  # sanity: the sample really is guarded
        assert guard._guard_condition_matches(guard.CLASS_CLUSTER_WRITE_GUARD_CLAUSES, source), (
            source
        )

    # And a normal doc is NOT guarded on either side.
    normal = {'class_source': 'item_model', 'class_validated': False}
    assert not is_locked_class(normal)
    assert not guard._guard_condition_matches(guard.CLASS_CLUSTER_WRITE_GUARD_CLAUSES, normal)


def test_class_cluster_write_guard_intentionally_diverges_on_test_holdout() -> None:
    """W10 fix pass (Opus review 2026-09-28, lock-rule call-site m3): an
    unvalidated ``test_holdout`` item IS locked by ``is_locked_class``
    (its class must never be touched by an automated writer), but this
    clause list intentionally does NOT guard it -- this write is cluster
    PLACEMENT (cluster_id/cluster_distance), not a class write, so
    residual clustering may still assign a holdout item's cluster id.
    Pins the divergence the (now corrected) module comment documents,
    so a future accidental narrowing/widening of either side is caught."""
    holdout_unvalidated = {
        'class_source': 'item_model',
        'class_validated': False,
        'test_holdout': True,
    }
    assert is_locked_class(holdout_unvalidated)
    assert not guard._guard_condition_matches(
        guard.CLASS_CLUSTER_WRITE_GUARD_CLAUSES, holdout_unvalidated
    )

    # A validated import IS covered on both sides -- is_locked_class's
    # import branch requires class_validated=True, which this clause
    # list already guards generically (not a divergence).
    validated_import = {'class_source': 'external_label', 'class_validated': True}
    assert is_locked_class(validated_import)
    assert guard._guard_condition_matches(guard.CLASS_CLUSTER_WRITE_GUARD_CLAUSES, validated_import)


def test_class_cluster_write_guard_script_text_names_same_fields_as_predicate() -> None:
    """The painless source literally names the fields/values
    CLASS_CLUSTER_WRITE_GUARD_CLAUSES encodes -- a future edit to the
    clause list is guaranteed to keep the two in sync since the script is
    *rendered from* the list, but this pins the exact field/value
    vocabulary the brief calls out."""
    src = guard._guard_condition_painless(guard.CLASS_CLUSTER_WRITE_GUARD_CLAUSES)
    assert "ctx._source['class_validated'] == true" in src
    assert "ctx._source['class_excluded'] == true" in src
    assert "ctx._source['class_source'].contains('human')" in src


def test_class_cluster_write_guard_noops_a_human_owned_doc() -> None:
    """Direct behavioral proof (no painless engine needed — the noop
    decision is table-driven from CLASS_CLUSTER_WRITE_GUARD_CLAUSES, the
    exact list the script is rendered from)."""
    human_doc = {'cluster_id': 5, 'class_source': 'human', 'class_validated': True}
    assert guard._guard_condition_matches(guard.CLASS_CLUSTER_WRITE_GUARD_CLAUSES, human_doc)


def test_class_cluster_write_guard_applies_to_a_normal_doc() -> None:
    normal_doc = {'cluster_id': 5, 'class_source': 'item_model', 'class_validated': False}
    assert not guard._guard_condition_matches(guard.CLASS_CLUSTER_WRITE_GUARD_CLAUSES, normal_doc)


# ---------------------------------------------------------------------------
# _bulk_update_subids — chunking, refresh, and the cluster-moved guard
# ---------------------------------------------------------------------------


class _RecordingBulkOS:
    def __init__(self) -> None:
        self.bulk_calls: list[dict[str, Any]] = []
        self.refresh_calls: list[str] = []

        class _Indices:
            def __init__(self, outer: _RecordingBulkOS) -> None:
                self._outer = outer

            async def refresh(self, *, index: str) -> None:
                self._outer.refresh_calls.append(index)

        self.indices = _Indices(self)

    async def bulk(self, *, body: list[dict[str, Any]], refresh: Any = False) -> dict[str, Any]:
        self.bulk_calls.append({'body': body, 'refresh': refresh})
        return {'errors': False}


@pytest.mark.asyncio
async def test_bulk_update_subids_chunks_at_1000_actions_with_one_final_refresh() -> None:
    client = _RecordingBulkOS()
    updates = [(f'crop{i}', f'42{chr(97 + i % 26)}') for i in range(1500)]

    n = await refine._bulk_update_subids(client, updates, expected_cluster_id=42)

    assert n == 1500
    # 1500 updates at chunk_size=1000 -> two bulk() calls.
    assert len(client.bulk_calls) == 2
    assert len(client.bulk_calls[0]['body']) == 2000  # 1000 updates * 2 body entries
    assert len(client.bulk_calls[1]['body']) == 1000  # remaining 500 updates
    for call in client.bulk_calls:
        assert call['refresh'] is False
    # One explicit refresh at the very end, not per chunk.
    assert client.refresh_calls == [refine.items_index()]


@pytest.mark.asyncio
async def test_bulk_update_subids_script_noops_if_cluster_id_changed() -> None:
    client = _RecordingBulkOS()
    await refine._bulk_update_subids(client, [('crop1', '42a')], expected_cluster_id=42)
    action = client.bulk_calls[0]['body'][1]
    assert 'script' in action
    src = action['script']['source']
    assert "ctx._source['cluster_id'] != params.cid" in src
    assert "ctx.op = 'noop'" in src
    assert action['script']['params'] == {
        'cid': 42,
        'subid': '42a',
        'now': action['script']['params']['now'],
    }


@pytest.mark.asyncio
async def test_bulk_update_subids_logs_partial_errors_without_raising() -> None:
    class _ErrorBulkOS(_RecordingBulkOS):
        async def bulk(self, *, body: list[dict[str, Any]], refresh: Any = False) -> dict[str, Any]:
            self.bulk_calls.append({'body': body, 'refresh': refresh})
            return {
                'errors': True,
                'items': [
                    {'update': {'_id': 'crop1', 'status': 409, 'error': {'type': 'conflict'}}}
                ],
            }

    client = _ErrorBulkOS()
    # Must not raise.
    n = await refine._bulk_update_subids(client, [('crop1', '42a')], expected_cluster_id=42)
    assert n == 1
