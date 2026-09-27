"""Tests for the class/region history helpers."""

from __future__ import annotations

from typing import Any

import pytest

from src.services.curation.history import (
    MAX_HISTORY_ENTRIES,
    MAX_REGION_CHAIN_ENTRIES,
    merge_region_chain,
    normalize_region_chain_entry,
    record_class_history,
    region_chain_entry,
)


class TestRecordClassHistory:
    def test_no_class_id_returns_existing_unchanged(self):
        # First labeling of a brand-new crop: nothing to preserve.
        src: dict[str, object] = {}
        result = record_class_history(src, writer='ingest')
        assert result == []

    def test_no_class_id_preserves_existing_history(self):
        # Edge case: somehow history exists but class_id was cleared.
        # Don't lose history; just don't append.
        src = {
            'class_id': None,
            'class_id_history': [{'class_id': 5, 'writer': 'vlm'}],
        }
        result = record_class_history(src, writer='ingest')
        assert result == [{'class_id': 5, 'writer': 'vlm'}]

    def test_appends_entry_with_current_state(self):
        src = {
            'class_id': 47,
            'class_name': 'pickup_truck',
            'class_source': 'item_model',
            'label_source': '',
            'confidence': 0.91,
        }
        result = record_class_history(src, writer='ingest', now='2026-05-15T00:00:00+00:00')
        assert len(result) == 1
        entry = result[0]
        assert entry['class_id'] == 47
        assert entry['class_name'] == 'pickup_truck'
        assert entry['class_source'] == 'item_model'
        assert entry['confidence'] == 0.91
        assert entry['writer'] == 'ingest'
        assert entry['at'] == '2026-05-15T00:00:00+00:00'
        assert entry['class_validated'] is False

    def test_appended_entry_records_pre_write_validation_state(self):
        # Snapshotting a validated crop's history entry must carry
        # class_validated=true, not just default to false.
        src = {
            'class_id': 47,
            'class_name': 'pickup_truck',
            'class_source': 'human',
            'class_validated': True,
        }
        result = record_class_history(src, writer='class_merge')
        assert result[-1]['class_validated'] is True

    def test_appends_to_existing_history(self):
        src = {
            'class_id': 47,
            'class_name': 'pickup_truck',
            'class_source': 'vlm',
            'class_id_history': [
                {'class_id': 47, 'class_source': 'item_model', 'writer': 'ingest'},
            ],
        }
        result = record_class_history(src, writer='vlm_pipeline')
        assert len(result) == 2
        assert result[0]['class_source'] == 'item_model'
        assert result[1]['class_source'] == 'vlm'
        assert result[1]['writer'] == 'vlm_pipeline'

    def test_caps_at_max_entries(self):
        # Pre-populate at the cap, then append one more.
        history = [
            {'class_id': i, 'class_source': 'vlm', 'writer': f'w{i}'}
            for i in range(MAX_HISTORY_ENTRIES)
        ]
        src = {
            'class_id': 99,
            'class_source': 'human',
            'class_id_history': history,
        }
        result = record_class_history(src, writer='human')
        assert len(result) == MAX_HISTORY_ENTRIES
        # Newest entry is at the tail.
        assert result[-1]['class_id'] == 99
        # Oldest dropped (since no seed_backfill stub).
        assert result[0]['class_id'] == 1

    def test_caps_preserves_seed_backfill_origin(self):
        history = [{'class_id': -1, 'writer': 'seed_backfill'}]
        history += [
            {'class_id': i, 'class_source': 'vlm', 'writer': f'w{i}'}
            for i in range(MAX_HISTORY_ENTRIES)
        ]
        src = {
            'class_id': 99,
            'class_source': 'human',
            'class_id_history': history,
        }
        result = record_class_history(src, writer='human')
        assert len(result) == MAX_HISTORY_ENTRIES
        # Seed stays at index 0.
        assert result[0]['writer'] == 'seed_backfill'
        # Newest is at the tail.
        assert result[-1]['class_id'] == 99

    # =========================================================================
    # Dedupe — skip the append when class_id AND class_source are both
    # unchanged from the last recorded entry.
    # =========================================================================

    def test_no_append_when_class_unchanged(self):
        # Call twice with the same class_id/class_source. Before the fix,
        # the second call always appends, producing length 2.
        src = {'class_id': 47, 'class_source': 'item_model', 'class_id_history': []}
        history_after_first = record_class_history(src, writer='ingest')
        assert len(history_after_first) == 1

        src_second_call = dict(src)
        src_second_call['class_id_history'] = history_after_first
        history_after_second = record_class_history(src_second_call, writer='ingest')
        assert len(history_after_second) == 1
        assert history_after_second == history_after_first

    def test_append_when_source_changes_but_class_does_not(self):
        # The inverse of test_no_append_when_class_unchanged: class_id is
        # the same but class_source changed — must still append.
        src = {'class_id': 47, 'class_source': 'item_model', 'class_id_history': []}
        history_after_first = record_class_history(src, writer='ingest')
        assert len(history_after_first) == 1

        src_second_call = dict(src)
        src_second_call['class_source'] = 'vlm'
        src_second_call['class_id_history'] = history_after_first
        history_after_second = record_class_history(src_second_call, writer='vlm_pipeline')
        assert len(history_after_second) == 2
        assert history_after_second[0]['class_source'] == 'item_model'
        assert history_after_second[1]['class_source'] == 'vlm'

    def test_append_when_class_changes_but_source_does_not(self):
        # Same class_source, different class_id — must still append.
        src = {'class_id': 47, 'class_source': 'human', 'class_id_history': []}
        history_after_first = record_class_history(src, writer='human:label_crop')
        src_second_call = dict(src)
        src_second_call['class_id'] = 12
        src_second_call['class_id_history'] = history_after_first
        history_after_second = record_class_history(src_second_call, writer='human:label_crop')
        assert len(history_after_second) == 2
        assert history_after_second[0]['class_id'] == 47
        assert history_after_second[1]['class_id'] == 12

    # =========================================================================
    # No cap once class_validated=true.
    # =========================================================================

    def test_validated_crop_history_is_not_truncated_at_32(self):
        history = [
            {'class_id': i, 'class_source': 'vlm', 'writer': f'w{i}'}
            for i in range(MAX_HISTORY_ENTRIES)
        ]
        src = {
            'class_id': 99,
            'class_source': 'human',
            'class_validated': True,
            'class_id_history': history,
        }
        result = record_class_history(src, writer='human:label_crop')
        # Uncapped: MAX_HISTORY_ENTRIES existing entries + 1 new one.
        assert len(result) == MAX_HISTORY_ENTRIES + 1
        assert result[0]['class_id'] == 0
        assert result[-1]['class_id'] == 99

    def test_unvalidated_crop_history_still_capped(self):
        # Non-regression: the cap still applies when class_validated is
        # falsy/absent — only the validated cohort is exempt.
        history = [
            {'class_id': i, 'class_source': 'vlm', 'writer': f'w{i}'}
            for i in range(MAX_HISTORY_ENTRIES)
        ]
        src = {
            'class_id': 99,
            'class_source': 'vlm',
            'class_validated': False,
            'class_id_history': history,
        }
        result = record_class_history(src, writer='vlm_pipeline')
        assert len(result) == MAX_HISTORY_ENTRIES


class TestRegionChain:
    """``region_detector_chain`` entries are exactly ``<actor>:<event>``."""

    def test_entry_format(self):
        assert region_chain_entry('primary_detector', 'hit') == 'primary_detector:hit'
        assert (
            region_chain_entry('segmenter', 'sanity_reject:aspect')
            == 'segmenter:sanity_reject:aspect'
        )

    def test_normalizes_legacy_double_colon_timestamped_entries(self):
        assert (
            normalize_region_chain_entry('primary_detector::hit@2026-09-24T03:09:23+00:00')
            == 'primary_detector:hit'
        )
        assert (
            normalize_region_chain_entry('vlm_visible::yes@2026-09-24T03:09:21+00:00')
            == 'vlm_visible:yes'
        )
        # Canonical entries (incl. multi-part events) pass through untouched.
        for entry in ('vlm_visible:no', 'segmenter:sanity_reject:aspect', 'det:hit'):
            assert normalize_region_chain_entry(entry) == entry

    def test_merge_dedups_across_passes_and_heals_legacy(self):
        stored = [
            'primary_detector::hit@2026-09-24T03:09:23+00:00',
            'primary_detector::combined_verify_ok@2026-09-24T03:09:23+00:00',
        ]
        merged = merge_region_chain(
            stored, ['primary_detector:hit', 'primary_detector:combined_verify_ok']
        )
        assert merged == ['primary_detector:hit', 'primary_detector:combined_verify_ok']

    def test_merge_appends_new_in_order(self):
        merged = merge_region_chain(['a:miss'], ['b:hit', 'b:combined_verify_ok', 'b:hit'])
        assert merged == ['a:miss', 'b:hit', 'b:combined_verify_ok']

    def test_merge_caps_drop_oldest(self):
        chain = [f'det{i}:hit' for i in range(MAX_REGION_CHAIN_ENTRIES)]
        merged = merge_region_chain(chain, ['segmenter:hit'])
        assert len(merged) == MAX_REGION_CHAIN_ENTRIES
        assert merged[-1] == 'segmenter:hit'
        assert 'det0:hit' not in merged


# =============================================================================
# auto_promote_clusters missing-writer regression.
#
# Other missing-writer cases that depend on modules not covered here —
# the region-detection worker's bulk writer, the class-merge writer,
# and the label-import writer — are not exercised in this file.
# =============================================================================


class _FakeAutoPromoteOS:
    """Enough of the AsyncOpenSearch surface for auto_promote_clusters:
    one aggregation search, one scroll-init search, one scroll page,
    then per-doc OCC get/update via occ_skip_on_conflict_bulk."""

    def __init__(self) -> None:
        self.update_calls: list[dict[str, Any]] = []

    async def search(self, *, index: str, body: dict[str, Any], **kw: Any) -> dict[str, Any]:  # noqa: ARG002
        if 'aggs' in body:
            return {
                'aggregations': {
                    'clusters': {
                        'buckets': [
                            {
                                # Composite-agg bucket key is a dict.
                                'key': {'cluster_id': 1},
                                'doc_count': 5,
                                'top_class': {
                                    'buckets': [{'key': 'class_a', 'doc_count': 5}],
                                },
                            },
                        ],
                    },
                },
            }
        # Scroll-init search for doc ids matching the promote query.
        return {
            '_scroll_id': 'scroll-1',
            # The promote scroll reads the class state it votes on.
            'hits': {'hits': [{'_id': 'crop-a', '_source': dict(_AUTO_PROMOTE_SOURCE)}]},
        }

    async def scroll(self, *, scroll_id: str, **kw: Any) -> dict[str, Any]:  # noqa: ARG002
        return {'_scroll_id': scroll_id, 'hits': {'hits': []}}

    async def clear_scroll(self, *, scroll_id: str, **kw: Any) -> dict[str, Any]:  # noqa: ARG002
        return {}

    async def mget(self, *, body: dict[str, Any]) -> dict[str, Any]:
        from curation.occ_fakes import make_mget_response

        found = {d['_id']: dict(_AUTO_PROMOTE_SOURCE) for d in body['docs']}
        return make_mget_response(found)

    async def bulk(self, *, body: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:  # noqa: ARG002
        from curation.occ_fakes import make_bulk_response, make_bulk_update_item

        items = []
        for action, doc in zip(body[0::2], body[1::2], strict=True):
            doc_id = action['update']['_id']
            self.update_calls.append(doc['doc'])
            items.append(make_bulk_update_item(doc_id, status=200))
        return make_bulk_response(items)


_AUTO_PROMOTE_SOURCE: dict[str, Any] = {
    'class_id': 7,
    'class_name': 'class_a',
    'class_source': 'item_model',
    'class_validated': False,
    'test_holdout': False,
}


async def _run_auto_promote_case() -> list[dict[str, Any]]:
    # Import via orchestrator's re-export, matching every real caller
    # (the pipeline router, the clusters router) — importing
    # auto_promote directly as the first cluster-related module in the
    # process can hit the pre-existing orchestrator<->auto_promote
    # circular import (see docs/design/curation_design_rationale.md §5
    # for why orchestrator.py is a large, ratchet-exempt file).
    from src.services.curation.clustering.orchestrator import auto_promote_clusters

    fake_os = _FakeAutoPromoteOS()
    result = await auto_promote_clusters(fake_os, min_purity=0.85, min_members=4, dry_run=False)
    assert result['promoted'] == 1
    assert len(fake_os.update_calls) == 1
    return fake_os.update_calls[0].get('class_id_history') or []


@pytest.mark.asyncio
async def test_auto_promote_appends_history() -> None:
    history = await _run_auto_promote_case()
    assert history, 'auto_promote: no class_id_history entry was written'
    assert history[-1]['writer'] == 'auto_promote'
    assert history[-1]['class_id'] == 7
    assert history[-1]['class_source'] == 'item_model'


# =============================================================================
# Curation worker combined-VLM path (bulk_writer.py's OCC merger)
#
# Exercises scripts/curation/worker/bulk_writer.py. Other class-writer
# cases (a class-merge router endpoint, a label-import script) are not
# covered in this file.
# =============================================================================


async def _run_curation_worker_case() -> list[dict[str, Any]]:
    from curation.occ_fakes import make_bulk_response, make_bulk_update_item, make_mget_response
    from scripts.curation.worker.bulk_writer import _bulk_update
    from scripts.curation.worker.state import _ItemTask
    from src.config import get_region_fields
    from src.services.curation.class_write_guard import class_state_token

    F = get_region_fields()
    t = _ItemTask(
        crop_id='crop-1',
        image_path='/dev/null/never-read',
        item_bbox_norm=(0.1, 0.1, 0.5, 0.5),
        region_status='pending',
        class_name='audi',
        group='cars',
    )
    t.update_doc = {
        'class_id': 9,
        'class_name': 'camaro',
        'class_source': 'vlm',
        'label_source': 'vlm',
        'class_validated': False,
        F.status: 'detected',
        F.bbox_norm: [0.2, 0.2, 0.3, 0.3],
    }

    source = {
        'class_id': 5,
        'class_name': 'class_a',
        'class_source': 'item_model',
        'class_validated': False,
        # Still in the pending state the task was fetched in.
        F.status: 'pending',
    }
    # ... and the class state it was fetched in.
    t.class_token = class_state_token(source)

    from unittest.mock import AsyncMock

    opensearch = AsyncMock()
    opensearch.mget = AsyncMock(return_value=make_mget_response({'crop-1': source}))
    opensearch.bulk = AsyncMock(
        return_value=make_bulk_response([make_bulk_update_item('crop-1', status=200)])
    )

    n_written, _n_skipped = await _bulk_update(opensearch, [t])
    assert n_written == 1
    assert opensearch.bulk.await_args is not None
    bulk_body = opensearch.bulk.await_args.kwargs['body']
    written_doc = bulk_body[1]['doc']
    return written_doc.get('class_id_history') or []


@pytest.mark.asyncio
async def test_curation_worker_appends_history() -> None:
    history = await _run_curation_worker_case()
    assert history, 'curation worker: no class_id_history entry was written'
    assert history[-1]['writer'] == 'region_worker'
    assert history[-1]['class_id'] == 5
    assert history[-1]['class_source'] == 'item_model'


# =============================================================================
# classes.py::merge_class (the writer originally deferred alongside the
# curation worker case above — its subject now exists on this tree).
# =============================================================================


class _FakeMergeOS:
    """Enough of the AsyncOpenSearch surface for merge_class: a holdout
    count (0 — merge is allowed to proceed), an update_by_query against
    the confirmed-labels index, a scroll over matching items, and the
    per-doc OCC get/update via occ_skip_on_conflict_bulk."""

    def __init__(self, *, source_validated: bool = False) -> None:
        self.update_calls: list[dict[str, Any]] = []
        self._source_validated = source_validated

    async def count(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG002
        return {'count': 0}

    async def update_by_query(
        self,
        *,
        index: str,  # noqa: ARG002
        body: dict[str, Any],  # noqa: ARG002
        **kw: Any,  # noqa: ARG002
    ) -> dict[str, Any]:
        return {'updated': 1}

    async def search(self, *, index: str, body: dict[str, Any], **kw: Any) -> dict[str, Any]:  # noqa: ARG002
        return {
            '_scroll_id': 'scroll-merge-1',
            'hits': {'hits': [{'_id': 'crop-b'}]},
        }

    async def scroll(self, *, scroll_id: str, **kw: Any) -> dict[str, Any]:  # noqa: ARG002
        return {'_scroll_id': scroll_id, 'hits': {'hits': []}}

    async def clear_scroll(self, *, scroll_id: str, **kw: Any) -> dict[str, Any]:  # noqa: ARG002
        return {}

    async def mget(self, *, body: dict[str, Any]) -> dict[str, Any]:
        from curation.occ_fakes import make_mget_response

        source = {
            'class_id': 3,
            'class_name': 'sedan',
            'class_source': 'item_model',
            'class_validated': self._source_validated,
            'test_holdout': False,
        }
        found = {d['_id']: source for d in body['docs']}
        return make_mget_response(found)

    async def bulk(self, *, body: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:  # noqa: ARG002
        from curation.occ_fakes import make_bulk_response, make_bulk_update_item

        items = []
        for action, doc in zip(body[0::2], body[1::2], strict=True):
            doc_id = action['update']['_id']
            self.update_calls.append(doc['doc'])
            items.append(make_bulk_update_item(doc_id, status=200))
        return make_bulk_response(items)


async def _run_class_merge_case_with_os(
    monkeypatch: pytest.MonkeyPatch, *, source_validated: bool = False
) -> tuple[dict[str, Any], _FakeMergeOS]:
    from types import SimpleNamespace

    import src.routers.curation.classes as classes_mod
    from src.routers.curation._class_models import ClassMergeRequest

    fake_reg = SimpleNamespace(
        merge_class=lambda source_id, target_id: {
            'source_id': source_id,
            'target_id': target_id,
            'deprecated': True,
            'source_name': 'sedan',
            'target_name': 'coupe',
        },
        get=lambda class_id: SimpleNamespace(class_name='coupe'),  # noqa: ARG005
    )
    monkeypatch.setattr(classes_mod, 'get_class_registry', lambda: fake_reg)

    fake_os = _FakeMergeOS(source_validated=source_validated)
    result = await classes_mod.merge_class(ClassMergeRequest(source_id=3, target_id=6), fake_os)
    assert result['deprecated'] is True
    assert len(fake_os.update_calls) == 1
    return result, fake_os


async def _run_class_merge_case(
    monkeypatch: pytest.MonkeyPatch, *, source_validated: bool = False
) -> list[dict[str, Any]]:
    _, fake_os = await _run_class_merge_case_with_os(monkeypatch, source_validated=source_validated)
    return fake_os.update_calls[0].get('class_id_history') or []


@pytest.mark.asyncio
async def test_merge_class_appends_history(monkeypatch: pytest.MonkeyPatch) -> None:
    history = await _run_class_merge_case(monkeypatch)
    assert history, 'class_merge: no class_id_history entry was written'
    assert history[-1]['writer'] == 'class_merge'
    assert history[-1]['class_id'] == 3
    assert history[-1]['class_source'] == 'item_model'


@pytest.mark.asyncio
async def test_merge_class_history_records_pre_merge_validation_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F-56: a human-validated crop merged into another class must have
    its pre-merge validated=true state on record — not just the class it
    came from — so an audit can tell it apart from a merely-suggested
    label."""
    history = await _run_class_merge_case(monkeypatch, source_validated=True)
    assert history[-1]['class_validated'] is True

    history_unvalidated = await _run_class_merge_case(monkeypatch, source_validated=False)
    assert history_unvalidated[-1]['class_validated'] is False


@pytest.mark.asyncio
async def test_merge_class_carries_over_validation_to_the_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F-56 follow-up (owner-aligned semantics): a merge must KEEP a
    human validation, not clear it — a human-validated crop of the source
    class stays validated=true under the target. class_id_history (see
    test above) separately remembers that the *prior* class was validated;
    this test is about the doc's own live class_validated field after the
    merge write, which used to be hardcoded to False regardless."""
    _, fake_os = await _run_class_merge_case_with_os(monkeypatch, source_validated=True)
    assert fake_os.update_calls[0]['class_validated'] is True

    _, fake_os_unvalidated = await _run_class_merge_case_with_os(
        monkeypatch, source_validated=False
    )
    assert fake_os_unvalidated.update_calls[0]['class_validated'] is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
