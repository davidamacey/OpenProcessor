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
one implementation. ``src.routers.curation.review.freeze_test_holdout`` and
any offline promotion tooling should both call :func:`select_test_holdout`
rather than reimplementing the sampling.

Freezing is high blast radius and permanent (crops never return to the
trainable pool), so every freeze also persists a durable, auditable record
via :func:`persist_freeze_record` — mirrors the snapshot-then-atomic-replace
convention used by ``ClassRegistry._atomic_write`` in
``src/clients/curation_opensearch.py`` so a bad freeze can be diagnosed and
reverted from the recorded crop-id list.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from src.config.curation import get_curation_config


# Every class that appears in the cohort gets at least this many crops held
# out (or all of them, if the class has fewer than this many validated
# crops) — see Appendix C Decision 1.
MIN_TEST_PER_CLASS = 5

# Composite-agg page size (max distinct (class_id, hdd_source) strata per
# page) and the per-stratum scan page size. Today's cohort (~380 human-
# validated crops across ~30 classes x 1 hdd_source) is nowhere near
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
# 500 pages x 1000/page = 500k crop_ids for one (class_id, hdd_source)
# stratum, ~1300x today's entire cohort (382 crops).
_STRATUM_SCAN_MAX_PAGES = 500


def build_cohort_query() -> dict[str, Any]:
    """The freeze cohort: human-validated crops only.

    Plan Phase 2 item 1 / Appendix C Decision 2 — the old
    ``label_source in ['v6_original_label', 'hdd_user_label']`` filter
    referenced values nothing in the repo ever writes (0 matches, live).
    ``class_source`` is mapped ``keyword`` directly on the live index — no
    ``.keyword`` subfield exists (queries against it were 400ing /
    silently matching nothing until this fix).
    """
    return {
        'bool': {
            'must': [
                {'term': {'class_validated': True}},
                {'term': {'class_source': 'human'}},
            ]
        }
    }


async def scan_stratum_crop_ids(
    opensearch: Any, index: str, cohort_query: dict[str, Any], class_id: int, hdd_source: str
) -> list[str]:
    """Real per-(class_id, hdd_source) scan for every matching ``crop_id``,
    via ``search_after`` — no OpenSearch result-window limit, unlike
    ``top_hits`` (which 400s past ``index.max_inner_result_window``, 100 by
    default, on any stratum bigger than that; see the module-level
    ``_STRATUM_SCAN_PAGE_SIZE`` comment above).
    """
    stratum_query = {
        'bool': {
            'must': [
                cohort_query,
                {'term': {'class_id': class_id}},
                {'term': {'hdd_source': hdd_source}},
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
        raise HTTPException(
            status_code=503,
            detail=(
                f'stratum (class_id={class_id}, hdd_source={hdd_source}) exceeded '
                f'{_STRATUM_SCAN_MAX_PAGES} scan pages; refusing to freeze a '
                'possibly-truncated sample'
            ),
        )
    return ids


async def fetch_cohort_strata(
    opensearch: Any, index: str, query: dict[str, Any]
) -> list[dict[str, Any]]:
    """Enumerate every ``(class_id, hdd_source)`` stratum in the
    cohort via composite agg (following ``after_key`` across pages), then
    real-scan each stratum for its full ``crop_id`` list.

    Plan Phase 2 item 2, three bugs in one call site:

    - ``hdd_source`` is mapped ``keyword`` directly on the live index — no
      ``.keyword`` subfield exists; querying one either 400s or (composite
      agg) silently returns nothing.
    - The original single ``size: 1000`` composite page silently dropped
      any strata beyond the first 1000 — never triggered in the current
      ~30-class cohort, but wrong. Follow ``after_key`` until it's absent.
    - The original per-bucket ``top_hits(size: 1000)`` cap 400s past
      OpenSearch's default ``index.max_inner_result_window`` (100) on any
      real stratum over 100 crops — replaced with a genuine per-stratum
      scan (:func:`scan_stratum_crop_ids`), matching the plan's "real
      per-stratum scan" alternative.

    Returns one dict per stratum: ``{'class_id': int, 'hdd_source': str,
    'crop_ids': list[str]}``.
    """
    buckets: list[dict[str, Any]] = []
    after_key: dict[str, Any] | None = None
    while True:
        composite: dict[str, Any] = {
            'size': _STRATA_PAGE_SIZE,
            'sources': [
                {'class_id': {'terms': {'field': 'class_id'}}},
                {'hdd_source': {'terms': {'field': 'hdd_source'}}},
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
            class_id = int(key.get('class_id') or -1)
            hdd_source = str(key.get('hdd_source') or '')
            crop_ids = await scan_stratum_crop_ids(opensearch, index, query, class_id, hdd_source)
            buckets.append({'class_id': class_id, 'hdd_source': hdd_source, 'crop_ids': crop_ids})
        after_key = strata.get('after_key')
        if not after_key or not page_buckets:
            break
    return buckets


def _default_state_dir() -> Path:
    """Read the configured curation state dir at call time (not import
    time) so tests can ``monkeypatch.setenv`` (via ``CurationConfig.from_env``)
    around a single call without import-order fragility."""
    return Path(get_curation_config().state_dir) / 'test_holdout'


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


def persist_freeze_record(
    *,
    crop_ids: list[str],
    sha: str,
    cohort_spec: dict[str, Any],
    per_class_counts: dict[str, int],
    state_dir: Path | None = None,
) -> Path:
    """Write a durable freeze record and return the snapshot path.

    Writes two files under ``state_dir`` (default
    ``CurationConfig.state_dir/test_holdout/``):

    - ``<ISO-timestamp>.json`` — an immutable snapshot of this freeze.
    - ``current.json`` — atomically replaced to always point at the latest
      freeze (tmp-write + fsync + rename, same durability contract as
      ``ClassRegistry._atomic_write``).

    Without this, ``test_holdout_sha`` in the API response is a number
    nobody can verify or use to revert a bad freeze (plan Phase 2, item 4).
    """
    target_dir = state_dir if state_dir is not None else _default_state_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(UTC)
    record = {
        'frozen_at': now.isoformat(),
        'test_holdout_sha': sha,
        'n_frozen': len(crop_ids),
        'cohort_spec': cohort_spec,
        'per_class_counts': per_class_counts,
        'crop_ids': sorted(crop_ids),
    }
    payload = json.dumps(record, indent=2)

    ts = now.strftime('%Y%m%dT%H%M%S%fZ')
    snapshot_path = target_dir / f'{ts}.json'
    snapshot_path.write_text(payload, encoding='utf-8')

    current_path = target_dir / 'current.json'
    tmp_path = target_dir / 'current.json.tmp'
    with tmp_path.open('w', encoding='utf-8') as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(current_path)

    return snapshot_path
