"""The embedding clauses and the wire/doc side of ``embedding_state``."""

from __future__ import annotations

from curation.query_fakes import matches
from src.config.curation import ITEM_EMBEDDING_FIELD
from src.routers.curation._item_models import ItemDoc
from src.services.curation.embedding_state import (
    EMBEDDED,
    FAILED,
    embedded_clause,
    not_embedded_clause,
)
from src.services.curation.item_doc import DetectedItem, build_item_doc
from src.services.curation.wire import serialize_item


def _doc(item: DetectedItem) -> dict:
    return build_item_doc(
        crop_id='c1',
        image_id='i1',
        image_path='/data/a.jpg',
        source='t',
        request_id='r',
        bbox_norm=[0.1, 0.1, 0.5, 0.5],
        item=item,
        now='2026-01-01T00:00:00Z',
        crop_area_norm=0.16,
        crop_rank_in_image=1,
        blur_full_var=None,
        blur_lap_var=None,
        blur_lap_ratio=None,
    )


def test_clauses_partition_the_items() -> None:
    with_vec = {ITEM_EMBEDDING_FIELD: [0.1]}
    without = {'crop_id': 'x'}
    assert matches(with_vec, embedded_clause())
    assert not matches(without, embedded_clause())
    assert matches(without, not_embedded_clause())
    assert not matches(with_vec, not_embedded_clause())


def test_clauses_take_another_field() -> None:
    assert embedded_clause('other') == {'exists': {'field': 'other'}}
    assert not_embedded_clause('other') == {'bool': {'must_not': [{'exists': {'field': 'other'}}]}}


def test_a_vector_always_means_embedded() -> None:
    item = DetectedItem((0, 0, 1, 1), 0.9, pe_embedding=[0.1], embedding_state=FAILED)
    assert _doc(item)['embedding_state'] == EMBEDDED


def test_a_declared_state_is_written_without_a_vector() -> None:
    assert (
        _doc(DetectedItem((0, 0, 1, 1), 0.9, embedding_state=FAILED))['embedding_state'] == FAILED
    )


def test_no_vector_and_no_declared_state_writes_no_state() -> None:
    assert 'embedding_state' not in _doc(DetectedItem((0, 0, 1, 1), 0.9))


def test_wire_carries_the_state_and_null_for_legacy() -> None:
    assert serialize_item({'crop_id': 'c', 'embedding_state': FAILED})['embedding_state'] == FAILED
    assert serialize_item({'crop_id': 'c'})['embedding_state'] is None


def test_item_doc_model_exposes_the_field() -> None:
    assert 'embedding_state' in ItemDoc.model_fields
