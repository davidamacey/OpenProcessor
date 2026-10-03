"""``ItemFilter`` -> clauses, evaluated for real by the in-memory index double."""

from __future__ import annotations

from typing import Any

import pytest

from curation.query_fakes import matches
from src.services.curation.item_filter import ItemFilter, item_filter_clauses, visibility_clauses


CORPUS: dict[str, dict[str, Any]] = {
    'person_det': {
        'proposal_name': 'person',
        'confidence': 0.9,
        'crop_area_norm': 0.30,
        'crop_rank_in_image': 1,
        'pe_embedding': [1.0],
        'embedding_state': 'embedded',
        'class_source': 'item_proposal',
    },
    'hotdog_det': {
        'proposal_name': 'hot dog',
        'confidence': 0.5,
        'crop_area_norm': 0.02,
        'crop_rank_in_image': 3,
        'embedding_state': 'not_selected',
    },
    'car_human': {
        'class_name': 'car',
        'proposal_name': 'car',
        'confidence': 0.7,
        'crop_area_norm': 0.10,
        'crop_rank_in_image': 2,
        'pe_embedding': [1.0],
        'embedding_state': 'embedded',
        'class_source': 'human',
        'class_validated': True,
    },
    'dog_import': {
        'proposal_name': 'dog',
        'confidence': 0.2,
        'crop_area_norm': 0.05,
        'crop_rank_in_image': 1,
        'embedding_state': 'failed',
        'import_ids': ['imp1'],
        'review_dismissed_at': '2026-01-01',
    },
    'sheep_sam3': {
        'class_name': 'sheep',
        'confidence': 0.6,
        'crop_area_norm': 0.2,
        'crop_rank_in_image': 2,
        'open_vocab_set': 'street',
        'source_prompt': 'a sheep',
        'embedding_state': 'deferred',
        'class_excluded': True,
    },
    'legacy_no_state': {'proposal_name': 'bird', 'confidence': 0.4, 'crop_area_norm': 0.01},
}


def select(**kw: Any) -> set[str]:
    clauses = item_filter_clauses(ItemFilter(**kw))
    query = {'bool': {'filter': clauses}}
    return {i for i, doc in CORPUS.items() if matches(doc, query)}


def test_empty_filter_selects_everything() -> None:
    assert item_filter_clauses(ItemFilter()) == []
    assert select() == set(CORPUS)


def test_class_names_match_proposal_or_class_name_normalized() -> None:
    assert select(class_names=['Hot-Dog', 'sheep']) == {'hotdog_det', 'sheep_sam3'}
    assert select(class_names=['car']) == {'car_human'}


def test_exclude_class_names() -> None:
    assert select(exclude_class_names=['person', 'hot_dog']) == set(CORPUS) - {
        'person_det',
        'hotdog_det',
    }


def test_confidence_range_is_inclusive() -> None:
    assert select(conf_min=0.5, conf_max=0.7) == {'hotdog_det', 'car_human', 'sheep_sam3'}


def test_box_size_range() -> None:
    assert select(min_area=0.05, max_area=0.2) == {'car_human', 'dog_import', 'sheep_sam3'}


def test_n_largest_per_image_is_the_rank_cap() -> None:
    assert select(max_rank=1) == {'person_det', 'dog_import'}


def test_origin_is_a_union_and_detector_is_the_remainder() -> None:
    assert select(origin=['import']) == {'dog_import'}
    assert select(origin=['human']) == {'car_human'}
    assert select(origin=['sam3']) == {'sheep_sam3'}
    assert select(origin=['detector']) == {'person_det', 'hotdog_det', 'legacy_no_state'}
    assert select(origin=['sam3', 'import']) == {'sheep_sam3', 'dog_import'}


def test_open_vocab_provenance_filters() -> None:
    assert select(open_vocab_set='street') == {'sheep_sam3'}
    assert select(source_prompt='a sheep') == {'sheep_sam3'}
    assert select(source_prompt='a goat') == set()


def test_embedding_state_embedded_means_has_a_vector() -> None:
    assert select(embedding_state=['embedded']) == {'person_det', 'car_human'}
    assert select(embedding_state=['not_selected', 'deferred']) == {'hotdog_det', 'sheep_sam3'}
    assert select(embedding_state=['failed']) == {'dog_import'}


def test_review_status() -> None:
    assert select(review_status=['validated']) == {'car_human'}
    assert select(review_status=['dismissed']) == {'dog_import'}
    assert select(review_status=['excluded']) == {'sheep_sam3'}
    assert select(review_status=['pending']) == {'person_det', 'hotdog_det', 'legacy_no_state'}


def test_predicates_combine_with_and() -> None:
    assert select(class_names=['person', 'car'], conf_min=0.8, embedding_state=['embedded']) == {
        'person_det'
    }


@pytest.mark.parametrize(
    'bad',
    [{'conf_min': 0.9, 'conf_max': 0.1}, {'min_area': 0.5, 'max_area': 0.1}, {'item_text': '--'}],
)
def test_malformed_bands_raise_value_error(bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=r'greater than|letter or digit'):
        item_filter_clauses(ItemFilter(**bad))


def test_visibility_defaults_hide_holdout_and_excluded() -> None:
    docs = {'a': {}, 'b': {'test_holdout': True}, 'c': {'class_excluded': True}}
    q = {'bool': {'filter': visibility_clauses(include_test=False, include_excluded=False)}}
    assert {k for k, d in docs.items() if matches(d, q)} == {'a'}
    assert visibility_clauses(include_test=True, include_excluded=True) == []
