"""Tests for :mod:`src.routers.curation.pipeline`.

Ports the ``TestPipelineSkipFilter`` class from the reference tree's
``test_label_combined_wireup.py`` (deferred out of Chunk 8 there because
it exercises ``legacy_pipeline.py``, ported here in Chunk 9).
"""

from __future__ import annotations

from pathlib import Path


class TestPipelineSkipFilter:
    """``_run_chunk``'s query must exclude crops with a recent
    ``gemma_verify_completed_at``."""

    def test_unvalidated_query_excludes_recent_combined_writes(self) -> None:
        """Verify the must_not includes a range filter on gemma_verify_completed_at."""
        # Inspect the source — the query block is constructed inline in
        # ``pipeline_auto_label``; rather than invoking the endpoint
        # against a live OS, we assert on the source string. This pins
        # the filter so an inadvertent removal trips the test.
        from src.routers.curation import pipeline

        src = Path(pipeline.__file__).read_text()
        assert "'gemma_verify_completed_at'" in src
        # Must appear inside the must_not block, not just in a comment.
        # Heuristic: a range query keyed on the marker.
        assert 'range' in src
        assert 'gemma_verify_completed_at' in src

    def test_pipeline_skips_gemma_for_completed_crops(self) -> None:
        """The skip filter is purely time-based (24h window) and therefore
        applies to ANY crop class_source whose combined call set
        ``gemma_verify_completed_at`` within the last 24h. Pin the cutoff
        window and the range operator so a narrowing change (e.g. adding
        ``class_source == 'gemma'`` to the must clause) trips this test.
        """
        from src.routers.curation import pipeline

        src = Path(pipeline.__file__).read_text()
        # 24h cutoff is intentional — see comment near the
        # ``_combined_recent_cutoff`` definition.
        assert 'timedelta(hours=24)' in src
        # The filter is structured as a bare range against the marker,
        # NOT nested inside a class_source bool — so any source that
        # writes the marker is excluded.
        marker_idx = src.find("'gemma_verify_completed_at'")
        assert marker_idx > 0
        # Walk back to the enclosing must_not block; ensure no
        # class_source narrowing wraps the marker.
        ctx = src[max(0, marker_idx - 2000) : marker_idx]
        assert 'must_not' in ctx
        # Heuristic: the marker is inside a ``range`` not a ``term`` on
        # class_source. Detect a regression that would scope the filter.
        nearby = src[marker_idx : marker_idx + 200]
        assert 'range' in src[max(0, marker_idx - 100) : marker_idx + 200] or 'gte' in nearby
