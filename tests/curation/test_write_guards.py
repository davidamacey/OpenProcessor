"""Write-path integrity tests (human-label guard, ``test_holdout`` guard).

A machine detector/labeler must never silently reclassify a crop a human
already labeled, and when it *does* write a class (a legitimate machine
write onto a non-human crop), it must never leave stale human-provenance
fields (``label_source='human'``, ``class_validated=true``) behind.

Every automated class writer must also exclude frozen ``test_holdout``
crops. The guard is scoped to CLASS fields only — region-field writes
stay unconditional (test_holdout protects class-label ground truth, not
region detection state).

``TestAutomatedClassWritersExcludeTestHoldout`` restores five (of the
reference file's seven) per-writer ``test_holdout``-exclusion checks
(plan Wave 5 T-2): the cascade writer's check already lives in
``TestShouldClassifyHoldoutGuard`` above, and the reference's
label-import lookup-query check targets a module never ported anywhere
in this plan (out of Wave 5's scope — that's Wave 2 territory). The
reference's two one-off-migration-script checks (``cleanup_low_conf_v6_labels``
/ class-id-realign) target operator tooling this plan never ports;
their live equivalents on this tree are the two ``must_not`` clauses in
``src/services/curation/probe_predictions.py``'s
``run_probe_inference``/``build_uncertainty_queue``, which is what those
two restored tests assert against instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from curation.occ_fakes import make_bulk_response, make_bulk_update_item, make_mget_response
from scripts.curation.worker.runner import _should_classify
from scripts.curation.worker.state import _ItemTask
from scripts.curation.worker.verify import _combined_class_update
from src.config import get_region_fields
from src.services.labeling.vlm_labeler import VlmCombinedReply


def _make_task(
    *,
    class_source: str = '',
    class_confidence: float = 0.0,
    class_validated: bool = False,
    test_holdout: bool = False,
) -> _ItemTask:
    return _ItemTask(
        crop_id='crop-1',
        image_path='/dev/null/never-read',
        vehicle_bbox_norm=(0.1, 0.1, 0.5, 0.5),
        plate_status='pending',
        class_name='audi',
        group='cars',
        class_source=class_source,
        class_confidence=class_confidence,
        class_validated=class_validated,
        test_holdout=test_holdout,
    )


# =============================================================================
# Human-label guard (_should_classify)
# =============================================================================


@pytest.mark.usefixtures('reference_ingest_profiles')
class TestShouldClassifyHumanGuard:
    def test_skips_human_sourced_crop(self) -> None:
        t = _make_task(class_source='human', class_validated=True)
        assert _should_classify(t, registry_loaded=True) is False

    def test_skips_human_move(self) -> None:
        t = _make_task(class_source='human_move', class_validated=True)
        assert _should_classify(t, registry_loaded=True) is False

    def test_skips_human_source_even_without_validated_flag(self) -> None:
        # Belt-and-suspenders: class_source alone (independent of
        # class_validated) must never let a human write through.
        t = _make_task(class_source='human', class_validated=False)
        assert _should_classify(t, registry_loaded=True) is False

    def test_skips_class_validated_true_regardless_of_source(self) -> None:
        t = _make_task(class_source='v6_model', class_confidence=0.2, class_validated=True)
        assert _should_classify(t, registry_loaded=True) is False

    def test_still_classifies_low_conf_v6_non_human_non_holdout(self) -> None:
        # Non-regression: the fix must not swallow the legitimate cohort
        # the worker exists to serve.
        t = _make_task(class_source='v6_model', class_confidence=0.3, class_validated=False)
        assert _should_classify(t, registry_loaded=True) is True

    def test_still_skips_high_conf_v6_cohort(self) -> None:
        t = _make_task(class_source='v6_model', class_confidence=0.9, class_validated=False)
        assert _should_classify(t, registry_loaded=True) is False

    def test_registry_not_loaded_always_skips(self) -> None:
        t = _make_task(class_source='coco_yolo11_proposal')
        assert _should_classify(t, registry_loaded=False) is False


# =============================================================================
# test_holdout guard, class-field scope only (_should_classify)
# =============================================================================


class TestShouldClassifyHoldoutGuard:
    def test_skips_test_holdout_crop(self) -> None:
        t = _make_task(class_source='v6_model', class_confidence=0.2, test_holdout=True)
        assert _should_classify(t, registry_loaded=True) is False

    def test_holdout_crop_with_coco_source_also_skipped(self) -> None:
        t = _make_task(class_source='coco_yolo11_proposal', test_holdout=True)
        assert _should_classify(t, registry_loaded=True) is False

    def test_non_holdout_crop_unaffected(self) -> None:
        t = _make_task(class_source='v6_model', class_confidence=0.2, test_holdout=False)
        assert _should_classify(t, registry_loaded=True) is True


# =============================================================================
# Write-path provenance reset (_combined_class_update)
# =============================================================================


class TestCombinedClassUpdateResetsProvenance:
    def test_resolved_class_resets_label_source_and_validated(self) -> None:
        reply = VlmCombinedReply(img_id='crop-1', class_id=0, class_confidence='high')
        update = _combined_class_update(reply, ['adventurebike'], name_to_id={'adventurebike': 1})
        assert update['class_source'] == 'vlm'
        assert update['label_source'] == 'vlm'
        assert update['class_validated'] is False

    def test_vlm_unmatched_also_resets_provenance(self) -> None:
        reply = VlmCombinedReply(img_id='crop-1', class_id=-1, class_confidence='low')
        update = _combined_class_update(reply, ['adventurebike'])
        assert update['class_source'] == 'vlm_unmatched'
        assert update['label_source'] == 'vlm'
        assert update['class_validated'] is False

    def test_classify_skipped_leaves_class_fields_untouched(self) -> None:
        # class_names=None/[] means the caller (via _should_classify)
        # decided NOT to ask the VLM to classify — e.g. a human-sourced
        # or test_holdout crop. No class_source/label_source/
        # class_validated key should appear at all, so the existing doc
        # is untouched.
        reply = VlmCombinedReply(img_id='crop-1', class_id=0, class_confidence='high')
        update = _combined_class_update(reply, None)
        assert 'class_source' not in update
        assert 'label_source' not in update
        assert 'class_validated' not in update

    def test_plate_fields_stay_unconditional_when_class_write_suppressed(self) -> None:
        """Region-write non-regression guard: proves the human/holdout
        guards did not leak into region data. make/model/plate_visible
        must still be written even when class fields are suppressed."""
        reply = VlmCombinedReply(
            img_id='crop-1',
            class_id=0,
            class_confidence='high',
            plate_visible=True,
            make='Honda',
            model='CBR',
        )
        update = _combined_class_update(reply, None)
        assert update[get_region_fields().visible] is True
        assert update['vlm_item_make'] == 'Honda'
        assert update['vlm_item_model'] == 'CBR'
        assert 'class_source' not in update


class TestAutomatedClassWritersExcludeTestHoldout:
    """One check per automated class writer, each against the writer's
    real query-building code (not a re-derived literal) so breaking the
    guard in the source actually fails the test."""

    @pytest.mark.asyncio
    async def test_auto_promote_scroll_query_has_holdout_must_not(self) -> None:
        # Import order matters: orchestrator.py imports auto_promote at the
        # bottom of its own file (an intentional, preserved circular
        # import — see auto_promote.py's module docstring), so importing
        # orchestrator first resolves it the same way the app does.
        import src.services.curation.clustering.auto_promote as auto_promote_mod
        import src.services.curation.clustering.orchestrator  # noqa: F401

        fake_client = AsyncMock()
        fake_client.search = AsyncMock(
            return_value={
                'aggregations': {
                    'clusters': {
                        'buckets': [
                            {
                                'key': 7,
                                'doc_count': 10,
                                'top_class': {
                                    'buckets': [{'key': 'sedan', 'doc_count': 9}],
                                },
                            },
                        ],
                    },
                },
            }
        )
        captured: dict[str, Any] = {}

        async def _fake_scroll_ids(_client: Any, *, index: str, query: dict[str, Any]) -> list[str]:
            captured['index'] = index
            captured['query'] = query
            return []

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(auto_promote_mod, '_scroll_ids', _fake_scroll_ids)
            await auto_promote_mod.auto_promote_clusters(fake_client, min_purity=0.5, min_members=1)

        assert captured, 'auto_promote_clusters never reached the scroll query'
        must_not = captured['query']['bool']['must_not']
        assert {'term': {'test_holdout': True}} in must_not

    @pytest.mark.asyncio
    async def test_classes_merge_query_has_holdout_must_not(self, tmp_path: Any) -> None:
        import src.routers.curation.classes as classes_mod
        from src.clients.curation_opensearch import ClassRegistry
        from src.routers.curation._common import ClassMergeRequest

        registry = ClassRegistry(path=tmp_path / 'class_registry.json')
        registry.add_class('sedan', group='vehicle')
        registry.add_class('suv', group='vehicle')
        classes = registry.load().classes
        source_id, target_id = classes[0].class_id, classes[1].class_id

        fake_os = AsyncMock()
        fake_os.count = AsyncMock(return_value={'count': 0})
        fake_os.update_by_query = AsyncMock(return_value={})
        fake_os.search = AsyncMock(return_value={'hits': {'hits': []}, '_scroll_id': None})

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(classes_mod, 'get_class_registry', lambda: registry)
            await classes_mod.merge_class(
                ClassMergeRequest(source_id=source_id, target_id=target_id), fake_os
            )

        assert fake_os.update_by_query.await_count >= 1
        query = fake_os.update_by_query.await_args_list[0].kwargs['body']['query']
        must_not = query['bool']['must_not']
        assert {'term': {'test_holdout': True}} in must_not

    @pytest.mark.asyncio
    async def test_probe_inference_query_excludes_holdout(self) -> None:
        import src.services.curation.probe_predictions as probe_mod

        fake_os = AsyncMock()
        fake_os.search = AsyncMock(return_value={'hits': {'hits': []}, '_scroll_id': None})

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                probe_mod,
                '_build_predictor',
                lambda *_a, **_kw: (lambda _crop: (None, 0.0, 0.0, 0.0), 'v0'),
            )
            processed = await probe_mod.run_probe_inference(
                model_path=Path('unused.onnx'),
                opensearch=fake_os,
            )

        assert processed == 0
        assert fake_os.search.await_args is not None
        query = fake_os.search.await_args.kwargs['body']['query']
        must_not = query['bool']['must_not']
        assert {'term': {'test_holdout': True}} in must_not

    @pytest.mark.asyncio
    async def test_uncertainty_queue_count_query_excludes_holdout(self) -> None:
        import src.services.curation.probe_predictions as probe_mod

        fake_os = AsyncMock()
        fake_os.count = AsyncMock(return_value={'count': 0})

        result = await probe_mod.build_uncertainty_queue(fake_os, percent=5.0)

        assert result == []
        assert fake_os.count.await_args is not None
        body = fake_os.count.await_args.kwargs['body']
        must_not = body['query']['bool']['must_not']
        assert {'term': {'test_holdout': True}} in must_not

    def test_vlm_label_batch_guard_predicate_skips_holdout_doc(self) -> None:
        """``vlm_label_batch`` fetches crops by explicit id (no query to
        filter), so its test_holdout guard is a per-doc predicate
        consulted inside the OCC merger before any class field is
        written — this is that predicate."""
        from src.routers.curation.vlm import _is_frozen_test_holdout

        assert _is_frozen_test_holdout({'test_holdout': True}) is True
        assert _is_frozen_test_holdout({'test_holdout': False}) is False
        assert _is_frozen_test_holdout({}) is False


class TestMergeClassRefusesFrozenCrops:
    @pytest.mark.asyncio
    async def test_merge_refuses_with_409_when_holdout_members_exist(self) -> None:
        import src.routers.curation.classes as classes_mod
        from src.routers.curation._common import ClassMergeRequest

        fake_os = AsyncMock()
        fake_os.count = AsyncMock(return_value={'count': 2})
        fake_os.update_by_query = AsyncMock()

        with pytest.raises(HTTPException) as exc_info:
            await classes_mod.merge_class(ClassMergeRequest(source_id=5, target_id=6), fake_os)
        assert exc_info.value.status_code == 409
        assert '2' in exc_info.value.detail
        fake_os.update_by_query.assert_not_called()


# =============================================================================
# occ.py reusable human-guard predicate + its consumers.
# `_should_classify` only covers the detection worker's combined-call
# path; `vlm_label_batch` takes caller-supplied crop_ids with NO
# upstream filter at all, so it needs its own per-crop check.
# =============================================================================


class TestIsHumanOwnedClassPredicate:
    def test_true_for_human_and_human_move(self) -> None:
        from src.clients.occ import is_human_owned_class

        assert is_human_owned_class({'class_source': 'human'}) is True
        assert is_human_owned_class({'class_source': 'human_move'}) is True

    def test_false_for_machine_sources(self) -> None:
        from src.clients.occ import is_human_owned_class

        assert is_human_owned_class({'class_source': 'vlm'}) is False
        assert is_human_owned_class({'class_source': 'v6_model'}) is False
        assert is_human_owned_class({}) is False


class TestStripClassWriteFields:
    def test_drops_class_fields_keeps_region_fields(self) -> None:
        from src.clients.occ import strip_class_write_fields

        update = {
            'class_id': 5,
            'class_source': 'vlm',
            'label_source': 'vlm',
            'class_validated': False,
            'region_status': 'detected',
            'region_bbox_norm': [0.1, 0.1, 0.2, 0.2],
        }
        stripped = strip_class_write_fields(update)
        assert 'class_id' not in stripped
        assert 'class_source' not in stripped
        assert stripped['region_status'] == 'detected'
        assert stripped['region_bbox_norm'] == [0.1, 0.1, 0.2, 0.2]


class TestVlmLabelBatchHumanGuard:
    @pytest.mark.asyncio
    async def test_human_owned_crop_never_reaches_vlm(self, tmp_path, monkeypatch) -> None:
        """The gap this guards against: ``vlm_label_batch`` fetches by
        caller-supplied crop_id with no upstream filter. Without the
        per-crop ``is_human_owned_class`` check, a human-owned crop's
        class_source/label_source could get silently overwritten by
        whatever the VLM returned.

        A real crop-cache JPEG is planted so the crop reaches the same
        "ready to send to the VLM" point the unguarded code would —
        without it, an unrelated jpeg-fetch failure would make this test
        pass for the wrong reason regardless of the guard.
        """
        import io

        from PIL import Image

        monkeypatch.setenv('GEMMA_CROP_CACHE_DIR', str(tmp_path))
        crop_id = 'human-crop-1'
        buf = io.BytesIO()
        Image.new('RGB', (64, 64), (10, 20, 30)).save(buf, format='JPEG')
        (tmp_path / f'{crop_id}.jpg').write_bytes(buf.getvalue())

        import src.routers.curation.vlm as vlm_mod
        from src.routers.curation.vlm import VlmLabelBatchRequest

        fake_labeler = AsyncMock()
        fake_labeler.label_or_propose_batch = AsyncMock(return_value=[])
        monkeypatch.setattr(vlm_mod, '_get_vlm_labeler', lambda *_a, **_k: fake_labeler)

        fake_reg = MagicMock()
        fake_reg.load = MagicMock(return_value=MagicMock(classes=[]))
        monkeypatch.setattr(vlm_mod, 'get_class_registry', lambda: fake_reg)

        fake_os = AsyncMock()
        fake_os.get = AsyncMock(
            return_value={
                '_source': {
                    'class_source': 'human',
                    'class_validated': True,
                    'image_path': '/dev/null/never-read.jpg',
                    'bbox_norm': [0.0, 0.0, 1.0, 1.0],
                }
            }
        )

        result = await vlm_mod.vlm_label_batch(VlmLabelBatchRequest(crop_ids=[crop_id]), fake_os)
        fake_labeler.label_or_propose_batch.assert_not_called()
        assert result == {'predicted': 0, 'updated': 0}


class TestPipelineVlmMergerHumanGuard:
    def test_predicate_covers_pipeline_human_write_shapes(self) -> None:
        """Deliberately NOT wired into pipeline.py's merger (unlike vlm.py
        and bulk_writer.py): adding the same belt-and-suspenders check
        there pushed the file close to the 700-LOC pre-commit gate
        (``scripts/codegen/check_file_size.py``), and it would only guard
        a window that provably doesn't exist — every human write path in
        this codebase (crops.py's ``_merge_label``/``_merge`` for
        move_crops) sets ``class_validated=True`` in the SAME atomic doc
        as ``class_source='human'``/``'human_move'``, and pipeline.py's
        own ``unvalidated_query`` already excludes ``class_validated=True``
        upstream — so a human-owned doc can never reach that merger in
        the first place. This test just pins the predicate's coverage of
        both real human-write shapes so that invariant stays true."""
        from src.clients.occ import is_human_owned_class

        assert is_human_owned_class({'class_source': 'human', 'class_validated': True}) is True
        assert is_human_owned_class({'class_source': 'human_move', 'class_validated': True}) is True


class TestDetectionWorkerBulkWriterHumanGuard:
    @pytest.mark.asyncio
    async def test_bulk_update_strips_class_fields_but_keeps_plate_fields(self) -> None:
        """Exercises the real ``_bulk_update`` (not a closure — module
        level in bulk_writer.py). A task whose update_doc carries BOTH
        class and region fields (the combined-call shape) must land only
        the region half on OpenSearch when the current doc is
        human-owned — region writes stay unconditional."""
        from scripts.curation.worker.bulk_writer import _bulk_update
        from scripts.curation.worker.state import _ItemTask

        t = _ItemTask(
            crop_id='crop-1',
            image_path='/dev/null/never-read',
            vehicle_bbox_norm=(0.1, 0.1, 0.5, 0.5),
            plate_status='pending',
            class_name='audi',
            group='cars',
        )
        t.update_doc = {
            'class_id': 9,
            'class_name': 'camaro',
            'class_source': 'vlm',
            'label_source': 'vlm',
            'region_status': 'detected',
            'region_bbox_norm': [0.2, 0.2, 0.3, 0.3],
        }

        opensearch = AsyncMock()
        opensearch.mget = AsyncMock(
            return_value=make_mget_response(
                {
                    'crop-1': {
                        'class_source': 'human',
                        'class_validated': True,
                        'region_status': 'pending',
                    }
                }
            )
        )
        opensearch.bulk = AsyncMock(
            return_value=make_bulk_response([make_bulk_update_item('crop-1', status=200)])
        )

        n_written, _n_skipped = await _bulk_update(opensearch, [t])
        assert n_written == 1
        assert opensearch.bulk.await_args is not None
        written_doc = opensearch.bulk.await_args.kwargs['body'][1]['doc']
        assert 'class_id' not in written_doc
        assert 'class_source' not in written_doc
        assert 'label_source' not in written_doc
        assert written_doc['region_status'] == 'detected'
        assert written_doc['region_bbox_norm'] == [0.2, 0.2, 0.3, 0.3]

    @pytest.mark.asyncio
    async def test_bulk_update_unaffected_for_non_human_current_doc(self) -> None:
        """Non-regression: the guard must not strip class fields for the
        common case (current doc is machine-sourced)."""
        from scripts.curation.worker.bulk_writer import _bulk_update
        from scripts.curation.worker.state import _ItemTask

        t = _ItemTask(
            crop_id='crop-1',
            image_path='/dev/null/never-read',
            vehicle_bbox_norm=(0.1, 0.1, 0.5, 0.5),
            plate_status='pending',
            class_name='audi',
            group='cars',
        )
        t.update_doc = {'class_id': 9, 'class_source': 'vlm', 'region_status': 'detected'}

        opensearch = AsyncMock()
        opensearch.mget = AsyncMock(
            return_value=make_mget_response(
                {
                    'crop-1': {
                        'class_source': 'v6_model',
                        'class_validated': False,
                        'region_status': 'pending',
                    }
                }
            )
        )
        opensearch.bulk = AsyncMock(
            return_value=make_bulk_response([make_bulk_update_item('crop-1', status=200)])
        )

        await _bulk_update(opensearch, [t])
        assert opensearch.bulk.await_args is not None
        written_doc = opensearch.bulk.await_args.kwargs['body'][1]['doc']
        assert written_doc['class_id'] == 9
