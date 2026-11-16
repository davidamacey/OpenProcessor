"""The acts of the differential test: what a person or a machine does to one item, and a seeded
generator of random sequences of them. An act is a tuple ``(name, *args)``; both worlds (today's
document and the claim fold) execute the same tuple."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable

Act = tuple[Any, ...]

CLASS_NAMES = ('car', 'truck', 'bus', 'van')
STATES = ('accepted', 'rejected', 'false_positive', 'proposed')
WHOLE_SET = ('detected', 'false_positive', 'verify_rejected', 'no_region_visible')
HUMAN_ACTS = (
    'label',
    'accept',
    'unlabel',
    'discard',
    'exclude',
    'restore',
    'review_dismiss',
    'review_undismiss',
    'vlm_dismiss',
    'probe_dismiss',
    'freeze',
    'region_status',
    'box_new',
    'box_state',
    'box_text',
    'box_move',
    'box_delete',
    'box_batch_state',
    'box_none',
    'geometry',
    'import_label',
)
MACHINE_ACTS = (
    'redetect',
    'vlm',
    'propagate',
    'open_vocab_target',
    'adopt',
    'classifier',
    'region_write',
)
BOX_A = (0.15, 0.15, 0.30, 0.30)
BOX_B = (0.32, 0.32, 0.48, 0.48)
BOX_C = (0.20, 0.35, 0.45, 0.48)
NEW_BOXES = (BOX_A, BOX_B, BOX_C)
GEOMETRIES = (
    (0.05, 0.05, 0.60, 0.60),
    (0.30, 0.30, 0.70, 0.70),
    (0.00, 0.10, 0.50, 0.50),
)


def random_act(rng: random.Random, *, machine_share: float = 0.4) -> Act:
    machine = rng.random() < machine_share
    name = rng.choice(MACHINE_ACTS if machine else HUMAN_ACTS)
    cls = rng.choice(CLASS_NAMES)
    makers: dict[str, Callable[[], Act]] = {
        'label': lambda: (name, cls),
        'discard': lambda: (name, rng.random() < 0.7, rng.random() < 0.6),
        'exclude': lambda: (name, rng.choice(('not_an_object', 'blurry'))),
        'region_status': lambda: (name, rng.choice(WHOLE_SET)),
        'box_new': lambda: (name, rng.choice(NEW_BOXES)),
        'box_state': lambda: (name, rng.randrange(3), rng.choice(STATES)),
        'box_text': lambda: (name, rng.randrange(3), rng.choice(('AB12', 'x', ''))),
        'box_move': lambda: (name, rng.randrange(3), rng.choice(NEW_BOXES)),
        'box_delete': lambda: (name, rng.randrange(3)),
        'box_batch_state': lambda: (
            name,
            rng.sample(range(3), rng.randrange(1, 3)),
            rng.choice(STATES),
        ),
        'geometry': lambda: (name, rng.choice(GEOMETRIES)),
        'import_label': lambda: (name, cls, rng.choice(('validated', 'suggestion'))),
        'redetect': lambda: (name, rng.choice((-0.03, 0.02, 0.04))),
        'vlm': lambda: (
            name,
            rng.choice(('matched', 'matched', 'unmatched', 'new_class_pending', 'empty')),
            cls,
        ),
        'propagate': lambda: (name, cls),
        'open_vocab_target': lambda: (name, cls),
        'adopt': lambda: (name,),
        'classifier': lambda: (name, cls, rng.random() < 0.3),
        'region_write': lambda: (
            name,
            rng.sample(NEW_BOXES, rng.randrange(1, 3)),
            rng.choice(('proposed', 'proposed', 'accepted')),
        ),
    }
    return makers.get(name, lambda: (name,))()


def random_sequence(seed: int, length: int = 8) -> tuple[str | None, list[Act]]:
    """``(starts with a classifier opinion, acts)`` for a seed. At most one import label per
    sequence (a second import of the same label is a no-op today and counts nothing here)."""
    rng = random.Random(seed)
    acts: list[Act] = []
    imported = False
    while len(acts) < length:
        act = random_act(rng)
        if act[0] == 'import_label':
            if imported:
                continue
            imported = True
        acts.append(act)
    return ('van' if rng.random() < 0.5 else None), acts
