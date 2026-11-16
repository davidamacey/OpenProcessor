"""Finding a claim's subject again: grouping, id, overlap, no match, frame change (15a 3.4)."""

from __future__ import annotations

import itertools
import random
from typing import Any

from src.domain import claim_builders as cb, claim_builders_region as cbr
from src.domain.claim_builders import ItemView
from src.domain.claim_match import BoxRef, DetectionRef, match_boxes, match_items
from src.domain.results import RegistryClass, RegistrySnapshot
from tests.domain.fold_support import BOX, FRAME, PHOTO, Log


def _id(ref: Any) -> int:
    assert ref is not None
    return int(ref.class_id)


def view(crop: str, bbox=BOX, *, frame: int = FRAME, machine=None) -> ItemView:
    return ItemView(crop_id=crop, photo_key=PHOTO, frame=frame, bbox=bbox, machine_bbox=machine)


def labelled(log: Log, crop: str, bbox=BOX, **kw) -> None:
    log.add(cb.label(view(crop, bbox, **kw), class_name='car', class_id_hint=1, op_id='op'))


def test_same_id_takes_the_group() -> None:
    log = Log()
    labelled(log, 'a')
    result = match_items(FRAME, [DetectionRef('a', (0.6, 0.6, 0.9, 0.9))], log.claims)
    assert result.matched == {'a': 'a'}  # the id wins even with no overlap


def test_overlap_takes_the_group_and_the_item_keeps_the_claim_id() -> None:
    log = Log()
    labelled(log, 'old')
    moved = (0.12, 0.10, 0.52, 0.50)
    result = match_items(FRAME, [DetectionRef('new', moved)], log.claims)
    assert result.matched == {'old': 'new'}
    assert result.new_detections == ()


def test_below_the_threshold_nothing_matches_and_an_asserting_claim_makes_the_item() -> None:
    log = Log()
    labelled(log, 'old')
    far = (0.60, 0.60, 0.95, 0.95)
    result = match_items(FRAME, [DetectionRef('new', far)], log.claims)
    assert result.matched == {}
    assert result.created == ('old',)
    assert result.new_detections == ('new',)


def test_a_detector_box_the_person_saw_also_matches() -> None:
    log = Log()
    shown = (0.30, 0.30, 0.70, 0.70)  # the person moved the box away from the detector's
    log.add(cb.label(view('old', shown, machine=BOX), class_name='car', class_id_hint=1, op_id='o'))
    result = match_items(FRAME, [DetectionRef('new', BOX)], log.claims)
    assert result.matched == {'old': 'new'}


def test_two_overlapping_detections_the_best_overlap_wins_the_other_stays_machine() -> None:
    log = Log()
    labelled(log, 'old')
    better = (0.10, 0.10, 0.50, 0.52)
    worse = (0.10, 0.10, 0.50, 0.66)
    result = match_items(FRAME, [DetectionRef('w', worse), DetectionRef('b', better)], log.claims)
    assert result.matched == {'old': 'b'}
    assert result.new_detections == ('w',)


def test_merge_two_claim_groups_one_detection_the_other_group_waits_or_is_created() -> None:
    log = Log()
    left = (0.10, 0.10, 0.50, 0.50)
    right = (0.12, 0.10, 0.52, 0.50)
    labelled(log, 'l', left)
    log.add(cb.review_dismiss(view('r', right)))  # a lone dismissal on the second group
    result = match_items(FRAME, [DetectionRef('d', left)], log.claims)
    assert result.matched == {'l': 'd'}
    assert result.dormant == ('r',)  # a dismissal never makes an object by itself
    assert result.created == ()


def test_split_leaves_the_claimed_box_and_two_machine_boxes() -> None:
    log = Log()
    labelled(log, 'old', (0.1, 0.1, 0.9, 0.5))
    halves = [DetectionRef('h1', (0.1, 0.1, 0.4, 0.5)), DetectionRef('h2', (0.6, 0.1, 0.9, 0.5))]
    result = match_items(FRAME, halves, log.claims)  # each half is below 0.5 of the claimed box
    assert result.matched == {}
    assert result.created == ('old',)
    assert result.new_detections == ('h1', 'h2')


def test_a_drawn_object_with_no_detection_is_created_from_its_claims() -> None:
    log = Log()
    log.add(
        cbr.draw_object(view('drawn'), class_name='car', class_id_hint=1, request_id='r', op_id='o')
    )
    assert match_items(FRAME, [], log.claims).created == ('drawn',)


def test_a_lone_dismissal_waits_for_a_later_detection() -> None:
    log = Log()
    log.add(cb.review_dismiss(view('gone')))
    assert match_items(FRAME, [], log.claims).dormant == ('gone',)
    later = match_items(FRAME, [DetectionRef('new', BOX)], log.claims)
    assert later.matched == {'gone': 'new'}  # a deleted object stays deleted


def test_a_frame_change_matches_nothing_and_is_reported() -> None:
    log = Log()
    labelled(log, 'old', frame=1)
    result = match_items(6, [DetectionRef('old', BOX)], log.claims)
    assert result.frame_mismatch == ('old',)
    assert result.matched == {}
    assert result.created == ()  # fail closed: not even re-created
    assert result.dormant == ()


def test_withdrawn_claims_do_not_make_a_group() -> None:
    log = Log()
    (claim,) = log.add(cb.label(view('a'), class_name='car', class_id_hint=1, op_id='o'))
    log.add([cb.withdraw(claim.claim_id, view('a'), op_id=None)])
    result = match_items(FRAME, [DetectionRef('a', BOX)], log.claims)
    assert result.matched == {}
    assert result.new_detections == ('a',)


def test_matching_is_independent_of_input_order() -> None:
    log = Log()
    labelled(log, 'g1', (0.10, 0.10, 0.50, 0.50))
    labelled(log, 'g2', (0.11, 0.10, 0.51, 0.50))
    dets = [
        DetectionRef('d1', (0.10, 0.10, 0.50, 0.51)),
        DetectionRef('d2', (0.11, 0.10, 0.51, 0.51)),
    ]
    answers = {
        tuple(sorted(match_items(FRAME, list(d), list(c)).matched.items()))
        for d in itertools.permutations(dets)
        for c in (log.claims, list(reversed(log.claims)), random.Random(3).sample(log.claims, 2))
    }
    assert len(answers) == 1


def test_boxes_match_by_id_then_overlap_and_a_split_matches_nothing() -> None:
    machine = [BoxRef('b1', (0.1, 0.1, 0.4, 0.4)), BoxRef('b9', (0.5, 0.5, 0.9, 0.9))]
    assert match_boxes({'b1': [None], 'x': [(0.52, 0.5, 0.9, 0.9)]}, machine) == {
        'b1': 'b1',
        'x': 'b9',
    }
    split = [BoxRef('n1', (0.0, 0.0, 0.3, 0.5)), BoxRef('n2', (0.6, 0.0, 0.9, 0.5))]
    assert match_boxes({'old': [(0.0, 0.0, 0.9, 0.5)]}, split) == {}


def test_claims_under_an_old_id_join_the_box_another_group_took_by_id() -> None:
    # A re-run changed the id of the box a person edited, and they edited it again under the new
    # id: both groups are that one box, so they must not stand as two.
    machine = [BoxRef('b3', (0.2, 0.35, 0.45, 0.48))]
    claimed = {'b3': [(0.15, 0.15, 0.3, 0.3)], 'b1': [(0.2, 0.35, 0.45, 0.48)]}
    assert match_boxes(claimed, machine) == {'b3': 'b3', 'b1': 'b3'}


def test_registry_resolution_follows_the_hint_merges_and_names() -> None:
    reg = RegistrySnapshot(
        (
            RegistryClass(1, 'car', active=False, merged_into=2),
            RegistryClass(2, 'auto'),
            RegistryClass(5, 'truck'),
            RegistryClass(6, 'old truck', active=False),
        )
    )
    assert _id(reg.resolve('car', 1)) == 2  # merged: followed to the end
    assert _id(reg.resolve('truck', 5)) == 5
    assert _id(reg.resolve('truck', 2)) == 5  # a hint of another name never selects its class
    assert _id(reg.resolve('Truck', None)) == 5  # identity is the normalised name
    assert reg.resolve('boat', 5) is None  # unknown name: unresolved, never a guess
    renamed = RegistrySnapshot((RegistryClass(5, 'lorry'), RegistryClass(9, 'truck')))
    assert _id(renamed.resolve('truck', 5)) == 9  # the renamed class is not captured by its old id
