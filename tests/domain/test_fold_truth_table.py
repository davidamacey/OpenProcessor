"""The truth table: every lock predicate of ``src/clients/occ_locks.py`` against the fold, on every
history of up to three acts from a small alphabet (15a 4.4).

The predicates are evaluated on today's document, and read off the folded state and its claims; a
difference must be one the plan decided (``tests/domain/world_allowed.py``)."""

from __future__ import annotations

import itertools

import pytest

from tests.domain.world_allowed import ALLOWED
from tests.domain.world_compare import run_acts


BOX = (0.32, 0.32, 0.48, 0.48)
ALPHABET = [
    ('label', 'car'),
    ('unlabel',),
    ('exclude', 'not_an_object'),
    ('restore',),
    ('freeze',),
    ('import_label', 'truck', 'validated'),
    ('geometry', (0.30, 0.30, 0.70, 0.70)),
    ('box_new', BOX),
    ('region_write', [(0.15, 0.15, 0.30, 0.30)], 'proposed'),
    ('box_state', 0, 'rejected'),
    ('box_text', 0, 'AB12'),
    ('region_status', 'false_positive'),
    ('vlm', 'matched', 'bus'),
    ('classifier', 'van', False),
    ('redetect', 0.03),
]
PREDICATES = (
    'p_is_locked_class',
    'p_is_human_deleted',
    'p_has_human_geometry',
    'p_has_import_geometry',
    'p_is_locked_geometry',
    'p_is_locked_item',
    'p_is_human_owned_item',
    'p_is_locked_box',
)


@pytest.mark.parametrize('classifier', [None, 'van'])
def test_every_predicate_matches_on_every_short_history(classifier: str | None) -> None:
    unexplained = []
    checked = 0
    for length in (1, 2, 3):
        for acts in itertools.product(ALPHABET, repeat=length):
            out = run_acts(classifier, list(acts), ALLOWED, predicates=True)
            checked += out.steps
            unexplained += [d for d in out.divergences if d.field in PREDICATES]
    assert checked > 5000
    assert not unexplained, [(d.field, d.doc, d.fold, d.acts) for d in unexplained[:5]]
