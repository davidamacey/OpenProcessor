"""Unit tests for ``scripts/curation/vlm_worker.py`` (F-11).

Two independent bugs: (1) the producer's in-flight guard didn't account
for the label-write route's ``refresh=False``, so a crop released right
before a fetch started could still be re-dispatched (duplicate GPU work);
(2) an empty ``classifier_class_sources()`` produced a dead
``terms: {class_source: []}`` clause instead of being omitted.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_SPEC = importlib.util.spec_from_file_location(
    'vlm_worker_module', _REPO_ROOT / 'scripts' / 'curation' / 'vlm_worker.py'
)
assert _SPEC is not None
assert _SPEC.loader is not None
vlm_worker = importlib.util.module_from_spec(_SPEC)
sys.modules['vlm_worker_module'] = vlm_worker
_SPEC.loader.exec_module(vlm_worker)


# =============================================================================
# _filter_fresh_ids -- the released_at guard
# =============================================================================


def test_filter_fresh_ids_excludes_in_flight() -> None:
    fresh = vlm_worker._filter_fresh_ids(
        ['a', 'b', 'c'], in_flight={'b'}, released_at={}, fetch_started=100.0
    )
    assert fresh == ['a', 'c']


def test_filter_fresh_ids_excludes_recently_released() -> None:
    """A fetch that started at or after a release must not immediately
    re-dispatch that id -- refresh=False means it might still see stale
    (pre-write) state."""
    fresh = vlm_worker._filter_fresh_ids(
        ['a', 'b'], in_flight=set(), released_at={'a': 100.0}, fetch_started=100.0
    )
    assert fresh == ['b']

    fresh = vlm_worker._filter_fresh_ids(
        ['a', 'b'], in_flight=set(), released_at={'a': 100.5}, fetch_started=100.0
    )
    assert fresh == ['b']


def test_filter_fresh_ids_includes_ids_released_before_the_fetch_started() -> None:
    fresh = vlm_worker._filter_fresh_ids(
        ['a', 'b'], in_flight=set(), released_at={'a': 50.0}, fetch_started=100.0
    )
    assert fresh == ['a', 'b']


def test_filter_fresh_ids_never_redispatches_the_same_crop_across_two_polls() -> None:
    """The load-bearing regression: a crop released by a consumer between
    two producer polls must appear in at most one poll's fresh set, even
    when the second poll's fetch started very close to the release."""
    in_flight: set[str] = {'crop-1'}
    released_at: dict[str, float] = {}

    # Poll 1: crop-1 is in flight, so it's excluded regardless of timing.
    poll1_fresh = vlm_worker._filter_fresh_ids(
        ['crop-1'], in_flight=in_flight, released_at=released_at, fetch_started=10.0
    )
    assert poll1_fresh == []

    # Consumer finishes and releases crop-1 at t=11 (refresh=False, so the
    # write may not be visible yet to a fetch already in flight).
    in_flight.discard('crop-1')
    released_at['crop-1'] = 11.0

    # Poll 2 started at t=10.5 -- before the release -- so it must NOT
    # re-dispatch crop-1 even though it's no longer in in_flight.
    poll2_fresh = vlm_worker._filter_fresh_ids(
        ['crop-1'], in_flight=in_flight, released_at=released_at, fetch_started=10.5
    )
    assert poll2_fresh == []

    # Poll 3, safely after the release, may dispatch it again.
    poll3_fresh = vlm_worker._filter_fresh_ids(
        ['crop-1'], in_flight=in_flight, released_at=released_at, fetch_started=12.0
    )
    assert poll3_fresh == ['crop-1']


# =============================================================================
# _build_pending_query -- empty classifier_class_sources() must not emit a
# dead terms:[] clause
# =============================================================================


def _has_terms_clause_with_empty_list(query: Any) -> bool:
    if isinstance(query, dict):
        for key, value in query.items():
            if key == 'terms' and isinstance(value, dict) and any(v == [] for v in value.values()):
                return True
            if _has_terms_clause_with_empty_list(value):
                return True
    elif isinstance(query, list):
        return any(_has_terms_clause_with_empty_list(v) for v in query)
    return False


def test_build_pending_query_omits_classifier_gate_when_sources_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        'src.services.curation.ingest_class_sources.classifier_class_sources', lambda: set()
    )
    query = vlm_worker._build_pending_query(0.8)
    assert not _has_terms_clause_with_empty_list(query)
    # No "classifier already confident" bool clause at all when the
    # source set is empty -- distinguishable from the non-empty case
    # below by the absence of any 'confidence' range clause in must_not.
    must_not = query['bool']['must_not']
    assert not any(
        'bool' in c and any('confidence' in str(inner) for inner in c['bool'].get('must', []))
        for c in must_not
    )


def test_build_pending_query_keeps_classifier_gate_when_sources_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        'src.services.curation.ingest_class_sources.classifier_class_sources',
        lambda: {'classifier'},
    )
    query = vlm_worker._build_pending_query(0.8)
    must_not = query['bool']['must_not']
    assert any(
        'bool' in c and {'terms': {'class_source': ['classifier']}} in c['bool'].get('must', [])
        for c in must_not
    )


def test_build_pending_query_warns_once_when_sources_empty(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        'src.services.curation.ingest_class_sources.classifier_class_sources', lambda: set()
    )
    monkeypatch.setattr(vlm_worker, '_classifier_sources_empty_warned', False)
    vlm_worker._build_pending_query(0.8)
    vlm_worker._build_pending_query(0.8)
    out = capsys.readouterr().out
    assert out.count('classifier_class_sources() is empty') == 1
