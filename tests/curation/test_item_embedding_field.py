"""Freshly ingested, unlabeled items must reach the review and VLM queues.

Both queues gate on "the item has an embedding". The gate once named a field
(``embedding``) that the items index never carries -- items store their
encoder vector in ``ITEM_EMBEDDING_FIELD`` -- so every fresh item silently
fell out of ``/review/all`` and the VLM worker's cohort.
"""

from __future__ import annotations

from typing import Any

from curation.query_fakes import matches
from src.config.curation import ITEM_EMBEDDING_FIELD
from src.services.curation.item_doc import DetectedItem, build_item_doc


def _fresh_unlabeled_doc() -> dict[str, Any]:
    item = DetectedItem(
        bbox_pixel=(10.0, 10.0, 110.0, 90.0),
        score=0.9,
        class_source='item_proposal',
        proposal_name='car',
        pe_embedding=[0.1] * 8,
    )
    return build_item_doc(
        crop_id='c1',
        image_id='i1',
        image_path='/data/a.jpg',
        source='test',
        request_id='r1',
        bbox_norm=[0.1, 0.1, 0.5, 0.5],
        item=item,
        now='2026-01-01T00:00:00Z',
        crop_area_norm=0.16,
        crop_rank_in_image=0,
        blur_full_var=None,
        blur_lap_var=None,
        blur_lap_ratio=None,
    )


def _bool(must: list[dict[str, Any]], must_not: list[dict[str, Any]]) -> dict[str, Any]:
    return {'bool': {'must': must, 'must_not': must_not}}


def test_item_doc_writes_the_item_embedding_field() -> None:
    doc = _fresh_unlabeled_doc()
    assert ITEM_EMBEDDING_FIELD in doc
    assert 'class_id' not in doc or doc['class_id'] is None


def test_fresh_unlabeled_item_is_in_review_all_tab() -> None:
    from src.services.curation.review_queries import build_tab_query

    must, must_not, _ = build_tab_query('all', include_test=True, text=None, max_rank=None)
    assert matches(_fresh_unlabeled_doc(), _bool(must, must_not))


def test_fresh_unlabeled_item_is_in_vlm_worker_cohort() -> None:
    from scripts.curation.vlm_worker import _build_pending_query

    assert matches(_fresh_unlabeled_doc(), _build_pending_query(0.8))


def test_pipeline_vlm_sweep_fetches_the_item_embedding() -> None:
    from src.routers.curation import pipeline

    assert ITEM_EMBEDDING_FIELD in pipeline.VLM_SWEEP_SOURCE_FIELDS


def test_vlm_worker_mirror_matches_config() -> None:
    from scripts.curation import vlm_worker

    assert vlm_worker.ITEM_EMBEDDING_FIELD == ITEM_EMBEDDING_FIELD
