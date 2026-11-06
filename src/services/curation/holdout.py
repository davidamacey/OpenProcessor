"""Test-holdout selection + persistence — the single canonical implementation.

This module resolves a class of bug where a service had two divergent
implementations of "which validated crops become the permanent test
holdout": one endpoint doing random per-(class_id, source) sampling seeded
by a request param, and a separate offline promotion script doing
SHA1-of-``crop_id`` deterministic ordering per ``class_id`` with a
``max(5, ...)`` floor.

The decision was to keep exactly one algorithm: the SHA1-deterministic one,
because it needs no seed to reproduce and guarantees every covered class
gets at least :data:`MIN_TEST_PER_CLASS` held-out crops. This module is that
one implementation. ``src.routers.curation.holdout_freeze.freeze_test_holdout`` and
any offline promotion tooling should both call :func:`select_test_holdout`
rather than reimplementing the sampling.

Freezing is high blast radius and permanent (crops never return to the
trainable pool), so every freeze also leaves a durable, auditable record: a
``holdout_freezes`` row (``src/db/repos/records.py``) and a snapshot blob of the
frozen crop-id list behind the storage seam (:func:`write_freeze_snapshot`), so a
bad freeze can be diagnosed and reverted from the recorded list.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from src.clients.storage.roots import build_rooted_store
from src.config.project_context import current_project, current_project_id
from src.config.storage import storage_root_paths
from src.core.exceptions import RequestRejectedError
from src.db.repos.records import HoldoutFreezeRow, forget_holdout_freeze, freeze_holdout
from src.db.session import project_tx
from src.domain.storage_keys import make_key
from src.services.curation.dataset_thresholds import MIN_TEST_CROPS_PER_CLASS


if TYPE_CHECKING:
    import uuid
    from collections.abc import AsyncIterator

    from sqlalchemy.ext.asyncio import AsyncSession


# Every class that appears in the cohort gets at least this many crops held
# out (or all of them, if the class has fewer than this many validated
# crops) — see Appendix C Decision 1.
MIN_TEST_PER_CLASS = MIN_TEST_CROPS_PER_CLASS

FreezeKind = Literal['curated', 'import']
# Where an import freeze's snapshots live, under the project's holdout area.
IMPORT_FREEZE_DIR = 'imports'

# Composite-agg page size (max distinct (class_id, source) strata per
# page) and the per-stratum scan page size. Today's cohort (~380 human-
# validated crops across ~30 classes x 1 source) is nowhere near
# either limit.
#
# A per-bucket ``top_hits`` was tried first and rejected: OpenSearch's
# default ``index.max_inner_result_window`` is 100, so any stratum over
# 100 crops (several exist live, e.g. class 18 has 193) 400s regardless of
# how the cap is chosen — raising the cap just moves the ceiling, it can't
# remove it without an index-settings change. Confirmed live against a
# disposable index during Phase 2 verification. A real per-stratum scan
# (``search_after``, no result-window limit) is correct at any size.
_STRATA_PAGE_SIZE = 1000
_STRATUM_SCAN_PAGE_SIZE = 1000
# Safety valve against a pathological/buggy infinite loop, not a real cap:
# 500 pages x 1000/page = 500k crop_ids for one (class_id, source)
# stratum, ~1300x today's entire cohort (382 crops).
_STRATUM_SCAN_MAX_PAGES = 500

# Sentinels for the composite agg's missing_bucket strata (a doc with no
# class_id or no source). class_id uses -1 (never a real class id);
# source uses an explicit string sentinel since '' was already a
# plausible (if unlikely) real value and would be ambiguous with "missing".
_MISSING_CLASS_ID_STRATUM = -1
_MISSING_HOLDOUT_SOURCE_STRATUM = '__none__'


def _equals_or_missing(field: str, value: Any, missing_sentinel: Any) -> dict[str, Any]:
    """``{'term': {field: value}}``, or (when ``value`` is the missing
    sentinel) a clause matching docs where ``field`` doesn't exist at all."""
    if value == missing_sentinel:
        return {'bool': {'must_not': [{'exists': {'field': field}}]}}
    return {'term': {field: value}}


def build_cohort_query() -> dict[str, Any]:
    """The freeze cohort: human-validated crops only.

    The retired
    ``label_source`` filter referenced values nothing in the repo ever
    writes (0 matches, live).
    ``class_source`` is mapped ``keyword`` directly on the live index — no
    ``.keyword`` subfield exists (queries against it were 400ing /
    silently matching nothing until this fix).
    """
    return {
        'bool': {
            'filter': [
                {'term': {'class_validated': True}},
                {'term': {'class_source': 'human'}},
            ]
        }
    }


async def scan_stratum_crop_ids(
    opensearch: Any, index: str, cohort_query: dict[str, Any], class_id: int, source: str
) -> list[str]:
    """Real per-(class_id, source) scan for every matching ``crop_id``,
    via ``search_after`` — no OpenSearch result-window limit, unlike
    ``top_hits`` (which 400s past ``index.max_inner_result_window``, 100 by
    default, on any stratum bigger than that; see the module-level
    ``_STRATUM_SCAN_PAGE_SIZE`` comment above).
    """
    stratum_query = {
        'bool': {
            'filter': [
                cohort_query,
                _equals_or_missing('class_id', class_id, _MISSING_CLASS_ID_STRATUM),
                _equals_or_missing('source', source, _MISSING_HOLDOUT_SOURCE_STRATUM),
            ]
        }
    }
    ids: list[str] = []
    search_after: list[Any] | None = None
    for _page in range(_STRATUM_SCAN_MAX_PAGES):
        body: dict[str, Any] = {
            'size': _STRATUM_SCAN_PAGE_SIZE,
            'query': stratum_query,
            'sort': [{'crop_id': 'asc'}, {'_id': 'asc'}],
            '_source': ['crop_id'],
        }
        if search_after is not None:
            body['search_after'] = search_after
        resp = await opensearch.search(index=index, body=body)
        hits = (resp.get('hits') or {}).get('hits', [])
        if not hits:
            break
        for h in hits:
            crop_id = (h.get('_source') or {}).get('crop_id')
            if crop_id:
                ids.append(crop_id)
        if len(hits) < _STRATUM_SCAN_PAGE_SIZE:
            break
        search_after = hits[-1].get('sort')
    else:
        raise RequestRejectedError(
            status_code=503,
            detail=(
                f'stratum (class_id={class_id}, source={source}) exceeded '
                f'{_STRATUM_SCAN_MAX_PAGES} scan pages; refusing to freeze a '
                'possibly-truncated sample'
            ),
        )
    return ids


async def fetch_cohort_strata(
    opensearch: Any, index: str, query: dict[str, Any]
) -> list[dict[str, Any]]:
    """Enumerate every ``(class_id, source)`` stratum in the
    cohort via composite agg (following ``after_key`` across pages), then
    real-scan each stratum for its full ``crop_id`` list.

    Three bugs fixed in this call site:

    - ``source`` is mapped ``keyword`` directly on the live index — no
      ``.keyword`` subfield exists; querying one either 400s or (composite
      agg) silently returns nothing.
    - The original single ``size: 1000`` composite page silently dropped
      any strata beyond the first 1000 — never triggered in the current
      ~30-class cohort, but wrong. Follow ``after_key`` until it's absent.
    - The original per-bucket ``top_hits(size: 1000)`` cap 400s past
      OpenSearch's default ``index.max_inner_result_window`` (100) on any
      real stratum over 100 crops — replaced with a genuine per-stratum
      scan (:func:`scan_stratum_crop_ids`).

    Returns one dict per stratum: ``{'class_id': int, 'source': str,
    'crop_ids': list[str]}``.
    """
    buckets: list[dict[str, Any]] = []
    after_key: dict[str, Any] | None = None
    while True:
        composite: dict[str, Any] = {
            'size': _STRATA_PAGE_SIZE,
            'sources': [
                # missing_bucket keeps docs with no class_id/source
                # in the strata enumeration (as an explicit null key)
                # instead of silently dropping them from the composite agg
                # entirely -- a doc missing one of these fields would
                # otherwise never be frozen into any stratum at all.
                {'class_id': {'terms': {'field': 'class_id', 'missing_bucket': True}}},
                {'source': {'terms': {'field': 'source', 'missing_bucket': True}}},
            ],
        }
        if after_key:
            composite['after'] = after_key
        body = {
            'size': 0,
            'query': query,
            'aggs': {'strata': {'composite': composite}},
        }
        resp = await opensearch.search(index=index, body=body)
        strata = (resp.get('aggregations') or {}).get('strata') or {}
        page_buckets = strata.get('buckets', [])
        for bucket in page_buckets:
            key = bucket.get('key') or {}
            # `int(key.get('class_id') or -1)` treated class_id == 0
            # the same as a missing key (`0 or -1` == -1 in Python),
            # silently misbucketing every class-0 crop as "unknown". Only
            # an actually-missing/None key (missing_bucket above) falls
            # back to the sentinel now.
            raw_class_id = key.get('class_id')
            class_id = int(raw_class_id) if raw_class_id is not None else _MISSING_CLASS_ID_STRATUM
            raw_source = key.get('source')
            source = str(raw_source) if raw_source is not None else _MISSING_HOLDOUT_SOURCE_STRATUM
            crop_ids = await scan_stratum_crop_ids(opensearch, index, query, class_id, source)
            buckets.append({'class_id': class_id, 'source': source, 'crop_ids': crop_ids})
        after_key = strata.get('after_key')
        if not after_key or not page_buckets:
            break
    return buckets


def select_test_holdout(
    crop_ids_by_class: dict[Any, list[str]],
    *,
    fraction: float = 0.2,
) -> tuple[list[str], dict[str, int]]:
    """Deterministic per-class stratified sample.

    For each ``class_id`` bucket, sort the candidate ``crop_id``\\ s by
    ``sha1(crop_id)`` and take ``max(MIN_TEST_PER_CLASS, round(len *
    fraction))``, capped at the bucket size. Purely deterministic — the
    same cohort always produces the same selection and the same sha, with
    no seed to record or lose.

    Args:
        crop_ids_by_class: ``class_id -> [crop_id, ...]`` for the eligible
            cohort (already filtered to ``class_validated=true AND
            class_source='human'`` by the caller).
        fraction: target holdout fraction per class, e.g. ``0.2`` for 20%.

    Returns:
        ``(chosen_crop_ids, per_class_counts)`` — ``chosen_crop_ids`` is
        sorted for determinism; ``per_class_counts`` maps ``str(class_id)``
        to the number chosen from that class.
    """
    chosen: list[str] = []
    per_class: dict[str, int] = {}
    for class_id, crop_ids in crop_ids_by_class.items():
        unique_ids = sorted({cid for cid in crop_ids if cid})
        if not unique_ids:
            continue
        # usedforsecurity=False: sha1 here is a deterministic shuffle key,
        # not a security boundary -- silences bandit B324.
        ordered = sorted(
            unique_ids,
            key=lambda cid: hashlib.sha1(cid.encode('utf-8'), usedforsecurity=False).hexdigest(),
        )
        n = max(MIN_TEST_PER_CLASS, round(len(ordered) * fraction))
        n = min(n, len(ordered))
        sample = ordered[:n]
        chosen.extend(sample)
        per_class[str(class_id)] = len(sample)
    return sorted(chosen), per_class


def compute_holdout_sha(crop_ids: list[str]) -> str:
    """``sha256`` over the newline-joined, sorted crop-id list.

    Sorting first makes the digest independent of iteration order.
    """
    return hashlib.sha256('\n'.join(sorted(crop_ids)).encode('utf-8')).hexdigest()


def write_freeze_snapshot(
    *,
    crop_ids: list[str],
    sha: str,
    cohort_spec: dict[str, Any],
    per_class_counts: dict[str, int],
    kind: FreezeKind = 'curated',
) -> str:
    """Write the snapshot blob of a freeze and return its storage key.

    ``projects/<slug>/holdout/<timestamp>.json`` for a curated freeze, under
    ``holdout/imports/`` for an import's or a combine's. Without it ``test_holdout_sha`` in the
    API response is a number nobody can verify or use to revert a bad freeze.
    """
    now = datetime.now(UTC)
    record = {
        'kind': kind,
        'frozen_at': now.isoformat(),
        'test_holdout_sha': sha,
        'n_frozen': len(crop_ids),
        'cohort_spec': cohort_spec,
        'per_class_counts': per_class_counts,
        'crop_ids': sorted(crop_ids),
    }
    name = f'{now.strftime("%Y%m%dT%H%M%S%fZ")}.json'
    slug = current_project().record.slug
    key = (
        make_key('projects', slug, 'holdout', IMPORT_FREEZE_DIR, name)
        if kind == 'import'
        else make_key('projects', slug, 'holdout', name)
    )
    build_rooted_store(storage_root_paths()).write_atomic(
        key, json.dumps(record, indent=2).encode('utf-8')
    )
    return key


async def record_standing_freeze(
    session: AsyncSession,
    *,
    crop_ids: list[str],
    cohort_spec: dict[str, Any],
    per_class_counts: dict[str, int],
    source: str,
    frozen_by: str,
) -> HoldoutFreezeRow:
    """Record a freeze that is not the project's curated split (an import's or a combine's).

    Writes the snapshot, then the ``holdout_freezes`` row with ``make_current=False``, so the
    curated freeze stays current. A database error propagates: nothing is recorded.
    """
    key = await asyncio.to_thread(
        write_freeze_snapshot,
        crop_ids=crop_ids,
        sha=compute_holdout_sha(crop_ids),
        cohort_spec=cohort_spec,
        per_class_counts=per_class_counts,
        kind='import',
    )
    project_id = current_project_id()
    async with project_tx(session, project_id):
        return await freeze_holdout(
            session,
            project_id,
            source=source,
            item_count=len(crop_ids),
            frozen_by=frozen_by,
            snapshot_storage_key=key,
            make_current=False,
        )


async def forget_freeze(session: AsyncSession, freeze_id: uuid.UUID) -> bool:
    """Drop the row of an undone import freeze (its items are no longer held out).

    The current freeze is never dropped; the snapshot blob stays as history. Returns whether a
    row was removed.
    """
    project_id = current_project_id()
    async with project_tx(session, project_id):
        return await forget_holdout_freeze(session, project_id, freeze_id)


@asynccontextmanager
async def freeze_transaction(
    session: AsyncSession, *, item_count: int, snapshot_storage_key: str
) -> AsyncIterator[HoldoutFreezeRow]:
    """The curated freeze: its row is written first and committed only if the block completes.

    The caller flags the items inside the block. A database error refuses the freeze before any
    item changes (fail closed); an error raised inside the block rolls the row back, so a record
    always covers exactly the flagged ids.
    """
    project_id = current_project_id()
    async with project_tx(session, project_id):
        yield await freeze_holdout(
            session,
            project_id,
            source='manual',
            item_count=item_count,
            frozen_by='api',
            snapshot_storage_key=snapshot_storage_key,
        )
