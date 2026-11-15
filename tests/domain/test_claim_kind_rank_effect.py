"""Which claim kinds can change the rank of an item among its photo's items (``RANK_EFFECT``).

``apply`` folds only the named items when no input can change rank, and trusts this table. The
completeness test makes a new kind in the catalogue fail until someone classifies it; the soundness test
runs the differential harness's random histories and checks that a kind classified as neutral never moved
an item's box area, the only thing rank is made of (``fold.rank_items``).
"""

from __future__ import annotations

import contextlib
import random

import pytest

from src.domain.claim_kinds import KINDS, RANK_CHANGING_KINDS, RANK_EFFECT, ClaimKind
from tests.domain.doc_world import RefusedError
from tests.domain.fold_world import FoldWorld
from tests.domain.world_acts import random_sequence


def test_every_kind_of_the_catalogue_is_classified_and_nothing_else() -> None:
    assert set(RANK_EFFECT) == set(ClaimKind)
    assert set(RANK_EFFECT) == set(KINDS)


def test_the_rank_changing_kinds_are_a_drawn_item_geometry_exclusion_and_a_withdraw() -> None:
    assert {
        ClaimKind.ITEM_DRAWN,
        ClaimKind.GEOMETRY_SET,
        ClaimKind.EXCLUDED,
        ClaimKind.RESTORED,
        ClaimKind.WITHDRAW,
    } == RANK_CHANGING_KINDS


@pytest.mark.parametrize(
    'kind',
    [
        ClaimKind.CLASS_SET,
        ClaimKind.CLASS_CLEARED,
        ClaimKind.HOLDOUT_FROZEN,
        ClaimKind.CLASS_MOVED,
        ClaimKind.REVIEW_DISMISSED,
        ClaimKind.VLM_DISMISSED,
        ClaimKind.NEEDS_NEW_CLASS,
        ClaimKind.BOX_CREATED,
        ClaimKind.BOX_MOVED,
        ClaimKind.BOX_STATE_SET,
        ClaimKind.BOX_TEXT_SET,
        ClaimKind.BOX_DELETED,
    ],
)
def test_a_label_a_dismissal_a_flag_a_move_and_a_box_claim_cannot_change_rank(
    kind: ClaimKind,
) -> None:
    assert kind not in RANK_CHANGING_KINDS


@pytest.mark.parametrize('seed', range(300))
def test_an_act_made_only_of_rank_neutral_claims_never_changes_the_box_area(seed: int) -> None:
    classifier, acts = random_sequence(seed, length=10)
    world = FoldWorld(classifier)
    for act in acts:
        if (
            act[0] == 'region_write'
        ):  # written by the differential harness, which owns the minted ids
            continue
        if act[0] == 'redetect':  # a detection result: classified rank changing by table
            world.apply(act)
            continue
        area, held = world.fold().state.area, len(world.log.claims)
        with contextlib.suppress(RefusedError):
            world.apply(act)
        kinds = {c.kind for c in world.log.claims[held:]}
        if kinds and not kinds & RANK_CHANGING_KINDS:
            assert world.fold().state.area == area, (seed, act, sorted(k.value for k in kinds))
        elif not kinds:
            assert world.fold().state.area == area, (seed, act)


def test_the_history_generator_exercises_both_classes_of_claim() -> None:
    rng_kinds: set[ClaimKind] = set()
    for seed in range(300):
        classifier, acts = random_sequence(seed, length=10)
        world = FoldWorld(classifier)
        for act in acts:
            if act[0] == 'region_write':
                continue
            with contextlib.suppress(RefusedError):
                world.apply(act)
        rng_kinds |= {c.kind for c in world.log.claims}
    assert rng_kinds & RANK_CHANGING_KINDS
    assert rng_kinds - RANK_CHANGING_KINDS
    assert random.Random(0)  # the seeds are fixed: a failure names its seed
