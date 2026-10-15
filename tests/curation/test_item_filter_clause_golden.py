"""Golden clause output of ``item_filter_clauses`` over a matrix of filter inputs.

The golden file was captured from the implementation before it was decomposed;
clause content AND order are pinned (clauses are AND-ed, but order is part of
the wire query the routes send).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from src.services.curation.item_filter import ItemFilter, item_filter_clauses


GOLDEN = Path(__file__).parent / 'golden' / 'item_filter_clauses_golden.json'

CASES: dict[str, dict[str, Any]] = {
    'empty': {},
    'class_names': {'class_names': ['person', 'hot dog']},
    'exclude_class_names': {'exclude_class_names': ['car']},
    'conf_min': {'conf_min': 0.3},
    'conf_max': {'conf_max': 0.8},
    'conf_band': {'conf_min': 0.3, 'conf_max': 0.8},
    'min_area': {'min_area': 0.1},
    'max_area': {'max_area': 0.5},
    'area_band': {'min_area': 0.1, 'max_area': 0.5},
    'area_equal': {'min_area': 0.2, 'max_area': 0.2},
    'max_rank': {'max_rank': 2},
    'origin_one': {'origin': ['human']},
    'origin_all': {'origin': ['detector', 'sam3', 'human', 'import']},
    'embedding_one': {'embedding_state': ['embedded']},
    'embedding_many': {'embedding_state': ['embedded', 'failed', 'deferred', 'not_selected']},
    'review_pending': {'review_status': ['pending']},
    'review_all': {'review_status': ['pending', 'validated', 'dismissed', 'excluded']},
    'source': {'source': 'upload'},
    'source_empty_string': {'source': ''},
    'class_id_zero': {'class_id': 0},
    'cluster_id': {'cluster_id': 7},
    'cluster_subid': {'cluster_subid': '7b'},
    'dataset_split': {'dataset_split': 'val'},
    'label_source': {'label_source': 'vlm'},
    'class_source': {'class_source': 'human'},
    'open_vocab_set': {'open_vocab_set': 'street'},
    'source_prompt': {'source_prompt': 'a sheep'},
    'import_id': {'import_id': 'imp1'},
    'label_validated_true': {'label_validated': True},
    'label_validated_false': {'label_validated': False},
    'proposed_true': {'proposed_by_import': True},
    'proposed_false': {'proposed_by_import': False},
    'negative_true': {'on_negative_frame': True},
    'negative_false': {'on_negative_frame': False},
    'gate_skipped_true': {'region_gate_skipped': True},
    'gate_skipped_false': {'region_gate_skipped': False},
    'needs_new_true': {'needs_new_class': True},
    'needs_new_false': {'needs_new_class': False},
    'dismissed_true': {'review_dismissed': True},
    'dismissed_false': {'review_dismissed': False},
    'blur': {'min_blur_ratio': 0.4},
    'classifier_low': {'classifier_conf_lt': 0.5},
    'item_text': {'item_text': 'red car'},
    'everything': {
        'class_names': ['person'],
        'exclude_class_names': ['car'],
        'conf_min': 0.1,
        'conf_max': 0.9,
        'min_area': 0.01,
        'max_area': 0.9,
        'max_rank': 3,
        'origin': ['human', 'import'],
        'embedding_state': ['embedded', 'failed'],
        'review_status': ['validated', 'pending'],
        'source': 's',
        'class_id': 1,
        'cluster_id': 2,
        'cluster_subid': '2a',
        'import_id': 'imp',
        'dataset_split': 'train',
        'label_source': 'ls',
        'class_source': 'cs',
        'label_validated': True,
        'proposed_by_import': False,
        'on_negative_frame': True,
        'needs_new_class': True,
        'review_dismissed': False,
        'min_blur_ratio': 0.2,
        'classifier_conf_lt': 0.6,
        'item_text': 'dog',
        'open_vocab_set': 'ov',
        'source_prompt': 'sp',
        'region_gate_skipped': True,
    },
}


def test_golden_covers_every_filter_field() -> None:
    used = {k for case in CASES.values() for k in case}
    assert used == set(ItemFilter.model_fields)


@pytest.mark.parametrize('name', sorted(CASES))
def test_clauses_match_golden(name: str) -> None:
    golden = json.loads(GOLDEN.read_text())
    assert item_filter_clauses(ItemFilter(**CASES[name])) == golden[name]


def test_golden_file_has_exactly_the_cases() -> None:
    assert set(json.loads(GOLDEN.read_text())) == set(CASES)


@pytest.mark.parametrize(
    ('kwargs', 'message'),
    [
        ({'min_area': 0.6, 'max_area': 0.5}, 'min_area (0.6) is greater than max_area (0.5)'),
        ({'item_text': '!!! ??'}, 'item_text must contain a letter or digit'),
    ],
)
def test_malformed_filter_raises(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=f'^{re.escape(message)}$'):
        item_filter_clauses(ItemFilter(**kwargs))


def test_conf_band_inverted_raises() -> None:
    with pytest.raises(ValueError, match='conf_min'):
        item_filter_clauses(ItemFilter(conf_min=0.9, conf_max=0.1))
