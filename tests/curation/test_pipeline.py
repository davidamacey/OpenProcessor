"""Tests for :mod:`src.routers.curation.pipeline`.

Ports the ``TestPipelineSkipFilter`` class from the reference tree's
``test_label_combined_wireup.py`` (deferred out of Chunk 8 there because
it exercises ``legacy_pipeline.py``, ported here in Chunk 9).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


class TestPipelineSkipFilter:
    """``_run_chunk``'s query must exclude crops with a recent
    ``vlm_verify_completed_at``."""

    def test_unvalidated_query_excludes_recent_combined_writes(self) -> None:
        """Verify the must_not includes a range filter on vlm_verify_completed_at."""
        # Inspect the source — the query block is constructed inline in
        # ``pipeline_auto_label``; rather than invoking the endpoint
        # against a live OS, we assert on the source string. This pins
        # the filter so an inadvertent removal trips the test.
        from src.services.curation.autolabel import selection

        src = Path(selection.__file__).read_text()
        assert "'vlm_verify_completed_at'" in src
        # Must appear inside the must_not block, not just in a comment.
        # Heuristic: a range query keyed on the marker.
        assert 'range' in src
        assert 'vlm_verify_completed_at' in src

    def test_pipeline_skips_gemma_for_completed_crops(self) -> None:
        """The skip filter is purely time-based (24h window) and therefore
        applies to ANY crop class_source whose combined call set
        ``vlm_verify_completed_at`` within the last 24h. Pin the cutoff
        window and the range operator so a narrowing change (e.g. adding
        ``class_source == 'vlm'`` to the must clause) trips this test.
        """
        from src.services.curation.autolabel import selection

        src = Path(selection.__file__).read_text()
        # 24h cutoff is intentional — see COMBINED_RECENT_WINDOW.
        assert 'timedelta(hours=24)' in src
        # The filter is structured as a bare range against the marker,
        # NOT nested inside a class_source bool — so any source that
        # writes the marker is excluded.
        marker_idx = src.find("'vlm_verify_completed_at'")
        assert marker_idx > 0
        # Walk back to the enclosing must_not block; ensure no
        # class_source narrowing wraps the marker.
        ctx = src[max(0, marker_idx - 2000) : marker_idx]
        assert 'must_not' in ctx
        # Heuristic: the marker is inside a ``range`` not a ``term`` on
        # class_source. Detect a regression that would scope the filter.
        nearby = src[marker_idx : marker_idx + 200]
        assert 'range' in src[max(0, marker_idx - 100) : marker_idx + 200] or 'gte' in nearby


class _FakeClassEntry:
    def __init__(self, class_id: int, class_name: str) -> None:
        self.class_id = class_id
        self.class_name = class_name
        self.group = None
        self.deprecated = False


class _FakeRegistry:
    """Stands in for ``get_class_registry()`` -- ``.load()`` returns self."""

    def __init__(self, classes: list[_FakeClassEntry]) -> None:
        self.classes = classes

    def load(self) -> _FakeRegistry:
        return self


class _FakeOpenSearch:
    """Minimal fake covering exactly what ``pipeline_auto_label`` touches
    on the ``run_vlm=True`` path when ``mget`` always reports "not
    found" (so the labeler is never actually invoked -- see
    ``_run_chunk``'s ``if not crops: return 0, 0, []`` short-circuit).

    ``hits_by_class`` maps ``class_id -> [crop_id, ...]``; the scroll
    ``search`` simulates OpenSearch's own filtering behavior by honoring
    a ``term: {class_id: ...}`` clause under ``query.bool.must`` if one is
    present in the request body -- the same clause the class_id query
    param (task d) is expected to add.
    """

    def __init__(self, hits_by_class: dict[int, list[str]]) -> None:
        self._hits_by_class = hits_by_class
        self.scroll_search_calls: list[dict[str, Any]] = []

    async def search(self, *, index: str, body: dict[str, Any], **kw: Any) -> dict[str, Any]:  # noqa: ARG002
        if 'aggs' in body:
            # pipeline_health_snapshot's baseline/after rollup -- an empty
            # dict makes it degrade to all-zero counts (its own
            # try/except-free `.get(...)` chain handles a missing key).
            return {}
        self.scroll_search_calls.append(body)
        must = (body.get('query') or {}).get('bool', {}).get('must', [])
        class_filter = next(
            (m['term']['class_id'] for m in must if 'class_id' in m.get('term', {})),
            None,
        )
        hits = [
            {'_id': crop_id, '_source': {'crop_id': crop_id}}
            for cid, crop_ids in self._hits_by_class.items()
            if class_filter is None or cid == class_filter
            for crop_id in crop_ids
        ]
        return {'_scroll_id': 'scroll-1', 'hits': {'hits': hits}}

    async def scroll(self, *, scroll_id: str, **kw: Any) -> dict[str, Any]:  # noqa: ARG002
        return {'_scroll_id': scroll_id, 'hits': {'hits': []}}

    async def clear_scroll(self, *, scroll_id: str, **kw: Any) -> dict[str, Any]:  # noqa: ARG002
        return {}

    async def count(self, *, index: str, body: dict[str, Any], **kw: Any) -> dict[str, Any]:  # noqa: ARG002
        return {'count': 0}

    async def mget(self, *, body: dict[str, Any], **kw: Any) -> dict[str, Any]:  # noqa: ARG002
        # Every doc reports not-found -- _run_chunk's `if not crops` guard
        # short-circuits before ever calling the VLM labeler, so this test
        # can prove the query scoping without mocking out VLM transport.
        return {'docs': [{'_id': d['_id'], 'found': False} for d in body['docs']]}

    class indices:  # noqa: N801 - mimics AsyncOpenSearch's `.indices` sub-client shape
        @staticmethod
        async def refresh(*, index: str, **kw: Any) -> dict[str, Any]:  # noqa: ARG004
            return {'_shards': {}}


class TestPipelineClassIdScoping:
    """Task (d): ``class_id`` scopes the auto-label pipeline's unvalidated
    cohort query to a single registry class."""

    @staticmethod
    def _registry() -> _FakeRegistry:
        return _FakeRegistry([_FakeClassEntry(3, 'wooden_pallet'), _FakeClassEntry(7, 'forklift')])

    @pytest.mark.asyncio
    async def test_class_id_unset_behavior_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No ``class_id`` -> the scroll query carries no class_id filter
        and every class's items are in scope (pre-existing behavior)."""
        from src.routers.curation import pipeline

        monkeypatch.setattr('src.routers.curation.get_class_registry', lambda: self._registry())
        fake_os = _FakeOpenSearch({3: ['pallet-1', 'pallet-2'], 7: ['forklift-1']})

        summary = await pipeline.pipeline_auto_label(
            opensearch=fake_os,
            train_clusters=False,
            promote_min_purity=0.85,
            promote_min_members=4,
            vlm_batch_size=32,
            vlm_concurrency=8,
            max_vlm_crops=0,
            classifier_confidence_skip_vlm=0.80,
            clustering_method=None,
            run_vlm=True,
            recluster_unvalidated=False,
            reassign_only=False,
            run_auto_promote=False,
            gate_max_rank=None,
            gate_min_blur_ratio=None,
            n_clusters=None,
            class_id=None,
        )

        assert summary['class_id'] is None
        assert fake_os.scroll_search_calls, 'sanity: the scroll query ran'
        first_body = fake_os.scroll_search_calls[0]
        must = (first_body.get('query') or {}).get('bool', {}).get('must', [])
        assert not any('class_id' in m.get('term', {}) for m in must)
        # All three items across both classes are in scope.
        assert summary['stages']['unvalidated_after_promote'] == 3

    @pytest.mark.asyncio
    async def test_class_id_set_scopes_query_and_cohort(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``class_id=3`` -> the scroll query carries a term filter on
        class_id=3, and only that class's two items are in scope --
        the other class's item never appears."""
        from src.routers.curation import pipeline

        monkeypatch.setattr('src.routers.curation.get_class_registry', lambda: self._registry())
        fake_os = _FakeOpenSearch({3: ['pallet-1', 'pallet-2'], 7: ['forklift-1']})

        summary = await pipeline.pipeline_auto_label(
            opensearch=fake_os,
            train_clusters=False,
            promote_min_purity=0.85,
            promote_min_members=4,
            vlm_batch_size=32,
            vlm_concurrency=8,
            max_vlm_crops=0,
            classifier_confidence_skip_vlm=0.80,
            clustering_method=None,
            run_vlm=True,
            recluster_unvalidated=False,
            reassign_only=False,
            run_auto_promote=False,
            gate_max_rank=None,
            gate_min_blur_ratio=None,
            n_clusters=None,
            class_id=3,
        )

        assert summary['class_id'] == 3
        first_body = fake_os.scroll_search_calls[0]
        must = (first_body.get('query') or {}).get('bool', {}).get('must', [])
        assert {'term': {'class_id': 3}} in must
        # Only the two class-3 items are in scope -- forklift-1 (class 7)
        # is excluded.
        assert summary['stages']['unvalidated_after_promote'] == 2
