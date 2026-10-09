"""Guarded bulk-write helpers for the item-cluster writers.

Shared by the refine path (``refine``) and the residual-pool clusterers
(``orchestrator``).
"""

from __future__ import annotations

from typing import Any

from src.core.logging import get_logger


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Guarded bulk writers.
#
# Every clustering writer below fetches candidates, spends seconds-to-minutes
# fitting a model, then bulk-writes cluster_id/cluster_subid back. A blind
# ``{'doc': {...}}`` update clobbers any human label/verification/exclusion
# made to a doc *during* that fit. Each writer instead sends a guarded
# painless ``script`` update that noops when the doc's current state shows
# human ownership -- the write simply doesn't happen; the doc keeps whatever
# the human set.
#
# A ``GuardClause`` list is the single source of truth for a guard: the same
# list renders the painless condition (``_guard_condition_painless``) and
# decides the noop in plain Python (``_guard_condition_matches``, used by
# tests), so the two can't independently drift the way hand-written painless
# text mirroring a separate Python predicate could.
GuardClause = tuple[str, str, Any]


def _guard_condition_painless(clauses: list[GuardClause]) -> str:
    """Render an OR-of-clauses guard condition as painless source.

    ``op='eq'`` -> ``ctx._source['field'] == <literal>``.
    ``op='contains'`` -> ``ctx._source['field'] != null &&
    ctx._source['field'].contains('<value>')``.

    Bracket notation throughout (not ``ctx._source.field``) so this stays
    correct even when a deployment renames a ``RegionFields`` attribute to
    something that isn't a valid painless identifier.
    """
    parts: list[str] = []
    for field, op, value in clauses:
        if op == 'eq':
            lit = 'true' if value is True else 'false' if value is False else f"'{value}'"
            parts.append(f"ctx._source['{field}'] == {lit}")
        elif op == 'contains':
            parts.append(
                f"(ctx._source['{field}'] != null && ctx._source['{field}'].contains('{value}'))"
            )
        else:
            raise ValueError(f'unsupported guard op {op!r}')
    return ' || '.join(parts)


def _guard_condition_matches(clauses: list[GuardClause], source: dict[str, Any]) -> bool:
    """Python-side mirror of :func:`_guard_condition_painless`.

    The same ``clauses`` list drives both, so a future change to one
    guard's fields/values automatically shows up on both sides -- there is
    nothing to keep "in sync" because there's only one definition.
    """
    for field, op, value in clauses:
        v = source.get(field)
        if op == 'eq' and v == value:
            return True
        if op == 'contains' and isinstance(v, str) and value in v:
            return True
    return False


# NOT a full mirror of src.clients.occ_locks.is_locked_class (W10 fix
# pass, Opus review 2026-09-28, lock-rule call-site m3): this covers the
# human-marker class_source check (a string containing 'human') plus the
# class_validated / class_excluded guards vlm.py's _class_locked already
# applies on its own write path, but it deliberately does NOT cover
# is_locked_class's `test_holdout` clause. This write is cluster
# PLACEMENT (cluster_id/cluster_distance), not a class write, so an
# unvalidated holdout item may still have its cluster assignment updated
# by residual clustering -- freezing a holdout item's CLASS is a
# separate guard (vlm.py, the region worker, class_write_guard.py), not
# this one. Kept as its own clause list (rather than calling
# is_locked_class from painless, which isn't possible);
# test_orchestrator_guarded_writes.py cross-checks the two stay
# equivalent on the fields this clause list DOES cover (human-marker,
# class_validated, class_excluded), with explicit holdout/validated-
# import samples pinning the intended divergence.
CLASS_CLUSTER_WRITE_GUARD_CLAUSES: list[GuardClause] = [
    ('class_validated', 'eq', True),
    ('class_excluded', 'eq', True),
    ('class_source', 'contains', 'human'),
]


def _guarded_class_cluster_write(cid: int, dist: float | None) -> dict[str, Any]:
    """Guarded bulk-update body for the residual/assign class-cluster
    writers (:func:`cluster_residuals`, :func:`assign_only_residuals`):
    noop instead of overwriting a human-owned (or validated/excluded)
    class row that changed while the fit was running."""
    return {
        'script': {
            'lang': 'painless',
            'params': {'cid': cid, 'dist': dist},
            'source': (
                f'if ({_guard_condition_painless(CLASS_CLUSTER_WRITE_GUARD_CLAUSES)})'
                " { ctx.op = 'noop'; return; }"
                " ctx._source['cluster_id'] = params.cid;"
                " ctx._source.remove('cluster_subid');"
                " ctx._source['cluster_distance'] = params.dist;"
                # The cluster the distance was measured against;
                # a later move leaves it pointing at the old cluster.
                " ctx._source['cluster_distance_cluster_id'] = params.cid;"
            ),
        }
    }


def _log_bulk_write_errors(op: str, resp: dict[str, Any]) -> None:
    """Log each failed bulk item (id + status + reason) at
    warning level instead of only a chunk-level 'errors: true' flag.
    Doesn't raise -- matches this module's existing partial-bulk-failure
    behavior of proceeding rather than aborting the whole run."""
    for item in resp.get('items') or []:
        action: dict[str, Any] = next(iter(item.values()), {})
        status = action.get('status')
        if status is not None and status >= 300:
            logger.warning(
                'clustering_bulk_write_item_failed',
                op=op,
                doc_id=action.get('_id'),
                status=status,
                error=action.get('error'),
            )
