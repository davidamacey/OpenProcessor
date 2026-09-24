"""F-3: the clustering orchestrator's bulk writers must never clobber a
concurrent human label/verification/exclusion.

Five writers fetch candidates, spend seconds-to-minutes fitting a model,
then bulk-write ``cluster_id``/``cluster_subid`` back. Before this fix each
one sent a blind ``{'doc': {...}}`` update, so a human write landing during
the fit was silently overwritten. Every writer now sends a guarded
painless ``script`` update instead: noop when the doc's current state
shows human ownership.

Two writers share :func:`_guarded_class_cluster_write` (vehicle-class
clustering: ``cluster_residuals``, ``assign_only_residuals``); two share
:func:`_guarded_region_write` (region clustering: ``cluster_region_residuals``,
``auto_assign_fp_from_centroids``); ``_bulk_update_subids`` (refine, both
domains) has its own "doc moved to a different cluster" guard.

The guard logic itself lives in a single ``GuardClause`` list per writer
kind (``CLASS_CLUSTER_WRITE_GUARD_CLAUSES`` /
``_region_write_guard_clauses``), rendered into painless text by
``_guard_condition_painless`` and evaluated in plain Python by
``_guard_condition_matches`` -- both read off the *same* list, so there is
nothing for a "predicate vs. script text" consistency test to catch
drifting apart; the tests below instead prove the list matches
``is_human_owned_class`` (class writes) and ``fp_candidate_must_not``
(region writes), and that the writers actually build ``script`` actions,
not ``doc`` actions.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.clients.occ import is_human_owned_class
from src.config import get_region_fields
from src.services.curation.clustering import orchestrator as orch


# ---------------------------------------------------------------------------
# Class-cluster write guard (cluster_residuals / assign_only_residuals)
# ---------------------------------------------------------------------------


def test_guarded_class_cluster_write_builds_a_script_action_not_doc() -> None:
    action = orch._guarded_class_cluster_write(10042, 0.12)
    assert 'script' in action
    assert 'doc' not in action
    assert action['script']['lang'] == 'painless'
    assert action['script']['params'] == {'cid': 10042, 'dist': 0.12}


def test_guarded_class_cluster_write_script_sets_expected_fields() -> None:
    src = orch._guarded_class_cluster_write(10042, 0.12)['script']['source']
    assert "ctx._source['cluster_id'] = params.cid" in src
    assert "ctx._source.remove('cluster_subid')" in src
    assert "ctx._source['cluster_distance'] = params.dist" in src
    assert "ctx.op = 'noop'" in src


def test_class_cluster_write_guard_matches_is_human_owned_class() -> None:
    """Cross-check: every source is_human_owned_class flags as human-owned
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
        human_owned = is_human_owned_class(source) or bool(
            source.get('class_validated') or source.get('class_excluded')
        )
        assert human_owned, source  # sanity: the sample really is guarded
        assert orch._guard_condition_matches(orch.CLASS_CLUSTER_WRITE_GUARD_CLAUSES, source), source

    # And a normal doc is NOT guarded on either side.
    normal = {'class_source': 'item_model', 'class_validated': False}
    assert not is_human_owned_class(normal)
    assert not orch._guard_condition_matches(orch.CLASS_CLUSTER_WRITE_GUARD_CLAUSES, normal)


def test_class_cluster_write_guard_script_text_names_same_fields_as_predicate() -> None:
    """The painless source literally names the fields/values
    CLASS_CLUSTER_WRITE_GUARD_CLAUSES encodes -- a future edit to the
    clause list is guaranteed to keep the two in sync since the script is
    *rendered from* the list, but this pins the exact field/value
    vocabulary the brief calls out."""
    src = orch._guard_condition_painless(orch.CLASS_CLUSTER_WRITE_GUARD_CLAUSES)
    assert "ctx._source['class_validated'] == true" in src
    assert "ctx._source['class_excluded'] == true" in src
    assert "ctx._source['class_source'].contains('human')" in src


def test_class_cluster_write_guard_noops_a_human_owned_doc() -> None:
    """Direct behavioral proof (no painless engine needed — the noop
    decision is table-driven from CLASS_CLUSTER_WRITE_GUARD_CLAUSES, the
    exact list the script is rendered from)."""
    human_doc = {'cluster_id': 5, 'class_source': 'human', 'class_validated': True}
    assert orch._guard_condition_matches(orch.CLASS_CLUSTER_WRITE_GUARD_CLAUSES, human_doc)


def test_class_cluster_write_guard_applies_to_a_normal_doc() -> None:
    normal_doc = {'cluster_id': 5, 'class_source': 'item_model', 'class_validated': False}
    assert not orch._guard_condition_matches(orch.CLASS_CLUSTER_WRITE_GUARD_CLAUSES, normal_doc)


# ---------------------------------------------------------------------------
# Region write guard (cluster_region_residuals / auto_assign_fp_from_centroids)
# ---------------------------------------------------------------------------


def test_guarded_region_write_builds_a_script_action_not_doc() -> None:
    F = get_region_fields()
    action = orch._guarded_region_write(F, {F.cluster_id: 3, F.cluster_subid: None})
    assert 'script' in action
    assert 'doc' not in action
    src = action['script']['source']
    assert f"ctx._source['{F.cluster_id}'] = params.v0" in src
    assert f"ctx._source.remove('{F.cluster_subid}')" in src


def test_region_write_guard_clauses_match_fp_candidate_must_not_human_terms() -> None:
    """fp_candidate_must_not() is the existing OpenSearch-query-side human
    guard for region writes; _region_write_guard_clauses is its bulk-write
    counterpart. Cross-check they name the same fields/values."""
    F = get_region_fields()
    query_clauses = orch.fp_candidate_must_not()
    query_terms = {
        (k, v) for c in query_clauses for k, v in c.get('term', {}).items() if k != F.status
    }
    assert (F.label_source, 'human') in query_terms
    assert (F.verifier, 'human') in query_terms

    write_clauses = orch._region_write_guard_clauses(F)
    write_terms = {(field, value) for field, op, value in write_clauses if op == 'eq'}
    assert (F.label_source, 'human') in write_terms
    assert (F.verifier, 'human') in write_terms
    # The write guard additionally noops on a human-*validated* region —
    # stricter than the FP-centroid query filter, which deliberately does
    # NOT exclude VLM-validated regions (see fp_candidate_must_not's
    # docstring) but a human validation is always final.
    assert (F.validated, True) in write_terms


def test_region_write_guard_noops_a_human_verified_doc() -> None:
    F = get_region_fields()
    human_doc = {F.verifier: 'human', F.label_source: 'model'}
    assert orch._guard_condition_matches(orch._region_write_guard_clauses(F), human_doc)


def test_region_write_guard_applies_to_a_normal_doc() -> None:
    F = get_region_fields()
    normal_doc = {F.verifier: 'vlm', F.label_source: 'model', F.validated: False}
    assert not orch._guard_condition_matches(orch._region_write_guard_clauses(F), normal_doc)


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

    n = await orch._bulk_update_subids(
        client, updates, cluster_id_field='cluster_id', expected_cluster_id=42
    )

    assert n == 1500
    # 1500 updates at chunk_size=1000 -> two bulk() calls.
    assert len(client.bulk_calls) == 2
    assert len(client.bulk_calls[0]['body']) == 2000  # 1000 updates * 2 body entries
    assert len(client.bulk_calls[1]['body']) == 1000  # remaining 500 updates
    for call in client.bulk_calls:
        assert call['refresh'] is False
    # One explicit refresh at the very end, not per chunk.
    assert client.refresh_calls == [orch.ITEMS_INDEX]


@pytest.mark.asyncio
async def test_bulk_update_subids_script_noops_if_cluster_id_changed() -> None:
    client = _RecordingBulkOS()
    await orch._bulk_update_subids(
        client, [('crop1', '42a')], cluster_id_field='cluster_id', expected_cluster_id=42
    )
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
    n = await orch._bulk_update_subids(
        client, [('crop1', '42a')], cluster_id_field='cluster_id', expected_cluster_id=42
    )
    assert n == 1
