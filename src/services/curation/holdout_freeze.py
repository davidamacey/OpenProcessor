"""The curated test-holdout freeze (``POST /test_holdout/freeze``) and the holdout stats.

Cohort and selection are :func:`~src.services.curation.holdout.build_cohort_query` and
:func:`~src.services.curation.holdout.select_test_holdout` (the single canonical algorithm). A
freeze is high blast radius and permanent, so it fails closed at every step: the guard that a
holdout already exists must be answerable, a database error refuses the freeze before any item
changes, and a failed flagging rolls the freeze record back.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.config.curation_accessors import items_index
from src.config.project_context import current_project_id
from src.core.exceptions import RequestRejectedError
from src.core.logging import get_logger
from src.db.session import project_tx
from src.services.curation.dataset_thresholds import MIN_TEST_CROPS_PER_CLASS
from src.services.curation.holdout import (
    MIN_TEST_PER_CLASS,
    build_cohort_query,
    compute_holdout_sha,
    fetch_cohort_strata,
    freeze_transaction,
    select_test_holdout,
    write_freeze_snapshot,
)
from src.services.near_duplicates.holdout import flag_held_out, leak_closure, record_closure


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


logger = get_logger(__name__)


class HoldoutExistsError(Exception):
    """A holdout is already frozen and ``force`` was not given."""


class HoldoutCheckError(Exception):
    """The count behind the "a holdout already exists" guard failed."""


class HoldoutSearchError(Exception):
    """Reading the freeze cohort from the items index failed."""


class HoldoutFlagError(Exception):
    """Flagging the held-out items failed (the freeze record is rolled back)."""


class EmptyHoldoutCohortError(Exception):
    """The selection froze nothing; a freeze that freezes nothing is never a success."""

    def __init__(self, cohort_size: int, n_classes: int) -> None:
        super().__init__(
            f'zero crops selected from a cohort of {cohort_size} across '
            f'{n_classes} classes; refusing to freeze nothing'
        )


@dataclass(frozen=True)
class FreezeResult:
    n_frozen: int
    moved_by_near_dup: int
    n_classes_covered: int
    test_holdout_sha: str
    per_class_counts: dict[str, int]
    percent: float
    min_per_class: int


async def _refuse_if_frozen(opensearch: Any) -> None:
    try:
        existing = await opensearch.count(
            index=items_index(), body={'query': {'term': {'test_holdout': True}}}
        )
    except Exception as exc:
        # Without the count the "a holdout already exists" guard cannot hold: refuse.
        logger.warning('curation_test_holdout_count_failed', error=str(exc))
        raise HoldoutCheckError from exc
    if (existing or {}).get('count', 0) > 0:
        raise HoldoutExistsError


async def freeze_test_holdout(
    opensearch: Any, session: AsyncSession, *, percent: float, force: bool
) -> FreezeResult:
    """Freeze a deterministic per-class ``percent`` of the human-validated items as the test set.

    Refuses to run if a holdout already exists unless ``force``. Persists a durable freeze
    record (a ``holdout_freezes`` row and a snapshot of the crop-id list); the row is written
    first inside the transaction that wraps the flagging. Raises :class:`HoldoutExistsError`,
    :class:`HoldoutCheckError`, :class:`HoldoutSearchError`, :class:`EmptyHoldoutCohortError`
    or :class:`HoldoutFlagError`; a database error propagates.
    """
    if not force:
        await _refuse_if_frozen(opensearch)

    cohort_query = build_cohort_query()
    try:
        strata = await fetch_cohort_strata(opensearch, items_index(), cohort_query)
    except Exception as exc:
        raise HoldoutSearchError from exc

    crop_ids_by_class: dict[int, list[str]] = {}
    for bucket in strata:
        ids = bucket.get('crop_ids') or []
        if not ids:
            continue
        crop_ids_by_class.setdefault(bucket['class_id'], []).extend(ids)

    fraction = percent / 100.0
    chosen_ids, per_class = select_test_holdout(crop_ids_by_class, fraction=fraction)

    if not chosen_ids:
        raise EmptyHoldoutCohortError(
            sum(len(v) for v in crop_ids_by_class.values()), len(crop_ids_by_class)
        )

    # Fail closed: a database error here refuses the freeze before anything is written.
    project_id = current_project_id()
    async with project_tx(session, project_id):
        closure = await leak_closure(session, project_id, chosen_ids)
    held_ids = sorted({*chosen_ids, *closure.moved_ids})

    digest = compute_holdout_sha(held_ids)
    cohort_spec = {
        'query': cohort_query,
        'selection': 'sha1_per_class',
        'percent': percent,
        'fraction': fraction,
        'force': force,
        'selected': len(chosen_ids),
        'moved_by_near_dup': len(closure.moves),
    }
    snapshot_key = await asyncio.to_thread(
        write_freeze_snapshot,
        crop_ids=held_ids,
        sha=digest,
        cohort_spec=cohort_spec,
        per_class_counts=per_class,
    )

    # Fail closed: the row is written first, inside the transaction that wraps the flagging.
    # A database error refuses the freeze before any item changes; a partial flagging raises
    # inside the block and rolls the row back, so a record always covers exactly the flagged
    # ids. A rerun is idempotent.
    async with freeze_transaction(
        session, item_count=len(held_ids), snapshot_storage_key=snapshot_key
    ) as freeze:
        try:
            await flag_held_out(opensearch, items_index(), held_ids)
        except RequestRejectedError:
            logger.error('curation_test_holdout_freeze_partial', n_ids=len(held_ids))
            raise
        except Exception as exc:
            raise HoldoutFlagError from exc
        await record_closure(session, project_id, freeze.id, closure)

    return FreezeResult(
        n_frozen=len(held_ids),
        moved_by_near_dup=len(closure.moves),
        n_classes_covered=len(per_class),
        test_holdout_sha=digest,
        per_class_counts=per_class,
        percent=percent,
        min_per_class=MIN_TEST_PER_CLASS,
    )


async def holdout_stats(opensearch: Any) -> dict[str, Any]:
    """Per-class test-set counts; ``deficient`` marks a class below the per-class minimum."""
    resp = await opensearch.search(
        index=items_index(),
        body={
            'size': 0,
            'query': {'term': {'test_holdout': True}},
            'aggs': {'by_class': {'terms': {'field': 'class_id', 'size': 1000}}},
            'track_total_hits': True,
        },
    )
    total = (resp.get('hits') or {}).get('total', {}).get('value', 0)
    buckets = (resp.get('aggregations') or {}).get('by_class', {}).get('buckets', [])
    return {
        'total': int(total),
        'by_class': [
            {**b, 'deficient': int(b.get('doc_count', 0)) < MIN_TEST_CROPS_PER_CLASS}
            for b in buckets
        ],
        'min_test_per_class': MIN_TEST_CROPS_PER_CLASS,
    }
