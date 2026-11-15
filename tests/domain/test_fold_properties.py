"""Property tests of the fold, each named after the review finding it guards (15a appendix A).

``hypothesis`` is not installed in the project venv, so the generators are seeded and the seeds
are fixed: a failure names its seed and reproduces. Every test here runs the pure fold only."""

from __future__ import annotations

import contextlib
import dataclasses
import random
from datetime import timedelta

import pytest

from src.domain import (
    claim_builders as cb,
    claim_builders_import as cbi,
    claim_builders_region as cbr,
)
from src.domain.claim_kinds import ClaimKind
from src.domain.claim_layers import split_claims
from src.domain.claim_match import DetectionRef, match_items
from src.domain.claims import without_replays
from src.domain.fold import fold_item, moved_placement
from src.domain.region_boxes import RegionBox
from src.domain.region_edit_rules import boxes_with_status
from src.domain.results import ItemRow, RegistryClass, RegistrySnapshot
from tests.domain.doc_world import RefusedError
from tests.domain.fold_support import (
    BOX,
    CROP,
    FRAME,
    PHOTO,
    REGISTRY,
    SETTINGS,
    VIEW,
    Log,
    detection,
    folded,
    inputs,
    machine_box,
    opinion,
    t,
)
from tests.domain.fold_world import FoldWorld
from tests.domain.world_acts import HUMAN_ACTS, random_act


SEEDS = range(400)
HUMAN_ONLY = tuple(a for a in HUMAN_ACTS if a != 'region_write')
CLASS_NAMES = ('car', 'truck', 'bus', 'van')


def human_world(seed: int, length: int = 6) -> FoldWorld:
    rng = random.Random(seed)
    world = FoldWorld('van' if seed % 2 else None)
    for _ in range(length):
        act = random_act(rng, machine_share=0.0)
        if act[0] not in HUMAN_ONLY:
            continue
        # a refused act (422 today) is simply not part of the history
        with contextlib.suppress(RefusedError):
            world.apply(act)
    return world


def machine_writes(world: FoldWorld, seed: int, count: int = 6) -> None:
    rng = random.Random(seed * 7919)
    for _ in range(count):
        act = random_act(rng, machine_share=1.0)
        if act[0] == 'region_write':
            box = RegionBox(box_id=f'b9{rng.randrange(9)}', bbox_norm=BOX, state='proposed')
            world.region_write_with([box], bump_seq=False)
            continue
        if act[0] == 'import_label':
            continue
        world.apply(act)


def comparable(fold):
    """The folded item without what only grows or moves with every write."""
    state = dataclasses.replace(
        fold.state, geometry_revision=0, region_revision=0, updated_at=t(0), rank=1
    )
    return state, fold.boxes


# ---- B1 -----------------------------------------------------------------------------------
@pytest.mark.parametrize('seed', SEEDS)
def test_pin_survives_machine_writes_b1(seed: int) -> None:
    world = human_world(seed)
    live = split_claims(world.log.claims).live
    pins = [
        c
        for c in live
        if c.kind
        in (
            ClaimKind.CLASS_SET,
            ClaimKind.CLASS_CLEARED,
            ClaimKind.HOLDOUT_FROZEN,
            ClaimKind.ITEM_DRAWN,
            ClaimKind.GEOMETRY_SET,
        )
    ]
    if not pins:
        pytest.skip('no pinning claim in this history')
    before = world.fold().state
    machine_writes(world, seed)
    after = world.fold().state
    assert after.bbox == before.bbox  # the box the person saw, whatever re-detected
    if before.class_claim_id is not None or before.holdout:
        assert (after.class_id, after.class_source, after.class_validated, after.class_name) == (
            before.class_id,
            before.class_source,
            before.class_validated,
            before.class_name,
        )


def test_a_redetect_overlapping_a_claimed_item_does_not_move_it_b1() -> None:
    log = Log()
    log.add(cb.label(VIEW, class_name='car', class_id_hint=1, op_id='o1'))
    shifted = (0.12, 0.10, 0.52, 0.50)
    before = folded(inputs(log.claims)).state
    after = folded(inputs(log.claims, detections=[detection(bbox=shifted, at=50)])).state
    assert before.bbox == after.bbox == BOX
    assert match_items(FRAME, [DetectionRef('new', shifted)], log.claims).matched == {CROP: 'new'}


# ---- B2 -----------------------------------------------------------------------------------
@pytest.mark.parametrize(
    'order', [(0, 1, 2), (2, 1, 0), (1, 2, 0), (1, 0, 2), (0, 2, 1), (2, 0, 1)]
)
def test_human_outranks_import_and_combine_b2(order: tuple[int, ...]) -> None:
    makers = [
        lambda: cb.label(VIEW, class_name='truck', class_id_hint=2, op_id='h'),
        lambda: cbi.import_label(
            VIEW, import_id='i1', class_name='car', class_id_hint=1, op_id='i'
        ),
        lambda: cbi.combine_label(VIEW, job_id='j1', class_name='bus', class_id_hint=3, op_id='c'),
    ]
    log = Log()
    for index in order:
        log.add(makers[index]())
    fold = folded(inputs(log.claims))
    assert (fold.state.class_name, fold.state.class_source) == ('truck', 'human')
    assert {c.kind for c in fold.conflicts} == {'locked_conflict', 'combine_conflict'}
    assert len(fold.conflicts) == 2


def test_the_newest_wins_inside_a_rank_b2() -> None:
    log = Log()
    log.add(cb.label(VIEW, class_name='car', class_id_hint=1, op_id='a'))
    log.add(cb.label(VIEW, class_name='bus', class_id_hint=3, op_id='b'))
    assert folded(inputs(log.claims)).state.class_name == 'bus'


# ---- B3 -----------------------------------------------------------------------------------
@pytest.mark.parametrize(
    'status', ['detected', 'false_positive', 'verify_rejected', 'no_region_visible']
)
@pytest.mark.parametrize('seed', range(30))
def test_whole_set_status_expands_b3(status: str, seed: int) -> None:
    rng = random.Random(seed)
    n = rng.randrange(1, 4)
    states = [rng.choice(['proposed', 'proposed', 'accepted', 'rejected']) for _ in range(n)]
    boxes = [
        machine_box(
            f'b{i + 1}',
            (0.1 * i, 0.1 * i, 0.1 * i + 0.08, 0.1 * i + 0.08),
            state=s,
            seq=1,
            rejection_reason='region_visible_elsewhere' if s == 'rejected' else None,
        )
        for i, s in enumerate(states)
    ]
    base = folded(inputs(boxes=boxes))
    shown_boxes = [
        RegionBox(b.box_id, b.bbox, b.state, rejection_reason=b.rejection_reason)
        for b in base.boxes
    ]
    try:
        expected = boxes_with_status(status, shown_boxes)
    except Exception:
        return
    log = Log()
    write = cbr.set_region_status(
        VIEW, shown_boxes, status, box_seq=len(boxes), region_seq=1, op_id='o'
    )
    log.add(write.drafts)
    machine_boxes = tuple(boxes)
    after = folded(inputs(log.claims, boxes=machine_boxes))
    assert [(b.box_id, b.state) for b in after.boxes] == [(b.box_id, b.state) for b in expected]
    assert after.state.region_validated is True
    # a machine box of a later run stays a result and is not shown in a closed set
    later = [*machine_boxes, machine_box('b50', (0.7, 0.7, 0.8, 0.8), seq=2)]
    closed = folded(inputs(log.claims, boxes=later))
    assert [b.box_id for b in closed.boxes] == [b.box_id for b in after.boxes]
    if status == 'no_region_visible':
        assert after.state.region_status == 'no_region_visible'
        assert after.boxes == ()


def test_partial_review_is_not_validated_b3() -> None:
    boxes = [machine_box('b1', (0.1, 0.1, 0.2, 0.2)), machine_box('b2', (0.5, 0.5, 0.6, 0.6))]
    base = folded(inputs(boxes=boxes))
    shown = [RegionBox(b.box_id, b.bbox, b.state) for b in base.boxes]
    log = Log()
    log.add(
        cbr.patch_box(
            VIEW, shown, 'b1', state='accepted', box_seq=2, region_seq=1, op_id='o'
        ).drafts
    )
    state = folded(inputs(log.claims, boxes=boxes)).state
    assert state.region_validated is False
    # a later machine run is still shown while the set is open
    newer = [*boxes, machine_box('b3', (0.7, 0.1, 0.8, 0.2), seq=2)]
    assert 'b3' in [b.box_id for b in folded(inputs(log.claims, boxes=newer)).boxes]


def test_box_deleted_never_creates_a_box_b3() -> None:
    log = Log()
    shown = [RegionBox('b1', (0.1, 0.1, 0.2, 0.2), 'proposed')]
    log.add(cbr.put_regions(VIEW, shown, [], box_seq=1, region_seq=1, op_id='o').drafts)
    state = folded(inputs(log.claims)).state  # no machine boxes at all
    assert state.region_count == 0
    assert folded(inputs(log.claims)).boxes == ()


# ---- B4 -----------------------------------------------------------------------------------
@pytest.mark.parametrize('seed', SEEDS)
def test_undo_of_any_subset_in_any_chunking_equals_the_fold_without_those_claims_b4(
    seed: int,
) -> None:
    rng = random.Random(seed)
    world = human_world(seed, length=7)
    claims = [c for c in world.log.claims if c.kind is not ClaimKind.WITHDRAW]
    if len(claims) < 2:
        pytest.skip('too short')
    chosen = rng.sample(claims, rng.randrange(1, len(claims) + 1))
    # the withdraws land in random chunks; the item is folded after every chunk
    log = Log(list(world.log.claims), clock=world.log.clock)
    pending = list(chosen)
    rng.shuffle(pending)
    results = []
    while pending:
        size = rng.randrange(1, 4)
        chunk, pending = pending[:size], pending[size:]
        log.add([cb.withdraw(claim.claim_id, VIEW, op_id='undo') for claim in chunk])
        results.append(
            fold_item(
                dataclasses.replace(world.inputs(), claims=tuple(log.claims)), REGISTRY, SETTINGS
            )
        )
    without = [c for c in world.log.claims if c.claim_id not in {x.claim_id for x in chosen}]
    expect = fold_item(
        dataclasses.replace(world.inputs(), claims=tuple(without)), REGISTRY, SETTINGS
    )
    last, want = results[-1], expect
    assert (want is None) == (last is None)
    if want is None or last is None:
        return
    holdout_kept = any(c.kind is ClaimKind.HOLDOUT_FROZEN for c in without)
    if holdout_kept:
        pytest.skip('a freeze keeps a withdrawn class claim on purpose (M1, tested apart)')
    assert comparable(last) == comparable(want)


# ---- M2 -----------------------------------------------------------------------------------
@pytest.mark.parametrize('seed', SEEDS)
def test_excluded_is_never_validated_m2(seed: int) -> None:
    world = human_world(seed, length=8)
    state = world.fold().state
    assert not (state.excluded and state.class_validated)


def test_restore_is_the_one_way_back_m2() -> None:
    log = Log()
    log.add(cb.label(VIEW, class_name='car', class_id_hint=1, op_id='l'))
    log.add(cb.exclude(VIEW, reason='blurry', op_id='e'))
    excluded = folded(inputs(log.claims)).state
    assert excluded.excluded
    assert excluded.class_validated is False
    log.add(cb.restore(VIEW, op_id='r'))
    restored = folded(inputs(log.claims)).state
    assert restored.excluded is False
    assert restored.class_validated is True  # computed from the class claim, no stored prior


# ---- M4 -----------------------------------------------------------------------------------
@pytest.mark.parametrize('seed', range(60))
def test_duplicate_chunk_replay_adds_nothing_m4(seed: int) -> None:
    rng = random.Random(seed)
    log = Log()
    drafts = cb.label(VIEW, class_name=rng.choice(CLASS_NAMES), class_id_hint=None, op_id='op')
    drafts += cb.exclude(VIEW, reason='blurry', op_id='op')
    log.add(without_replays(log.claims, drafts))
    first = folded(inputs(log.claims))
    assert without_replays(log.claims, drafts) == []  # the replay of the whole chunk
    log.add(without_replays(log.claims, drafts))
    assert len(log.claims) == 2
    assert comparable(folded(inputs(log.claims))) == comparable(first)
    # a later step-back undo still clears the item whole
    history = [cb.LiveOperation('op', 'label', tuple(c.claim_id for c in log.claims))]
    log.add(cb.unlabel(VIEW, history, op_id='undo'))
    assert folded(inputs(log.claims)).state.excluded is False


# ---- M1 -----------------------------------------------------------------------------------
def _freeze(log: Log, world_inputs, op: str = 'f'):
    from src.domain.fold_class import freeze_machine_class

    machine = freeze_machine_class(world_inputs, REGISTRY, SETTINGS)
    return log.add(cb.holdout_freeze(VIEW, freeze_id='f1', machine_class=machine, op_id=op))


def test_holdout_freezes_the_class_m1() -> None:
    opinions = [opinion('classifier', 'car', outcome='model')]
    log = Log()
    _freeze(log, inputs(log.claims, opinions=opinions))
    later = [*opinions, opinion('vlm', 'bus', at=500)]
    state = folded(inputs(log.claims, opinions=later)).state
    assert (state.class_name, state.holdout) == ('car', True)  # a later opinion cannot relabel it


def test_a_freeze_with_no_class_gets_none_from_later_opinions_m1() -> None:
    log = Log()
    _freeze(log, inputs(log.claims))
    state = folded(inputs(log.claims, opinions=[opinion('vlm', 'bus', at=500)])).state
    assert state.class_id is None


def test_a_withdraw_written_after_the_freeze_of_an_older_class_claim_changes_nothing_m1() -> None:
    log = Log()
    (label,) = log.add(cb.label(VIEW, class_name='truck', class_id_hint=2, op_id='l'))
    _freeze(log, inputs(log.claims))
    log.add([cb.withdraw(label.claim_id, VIEW, op_id='u')])
    assert folded(inputs(log.claims)).state.class_name == 'truck'  # kept_holdout


def test_a_persons_newer_claim_still_decides_a_frozen_item_m1() -> None:
    log = Log()
    log.add(cb.label(VIEW, class_name='truck', class_id_hint=2, op_id='l'))
    _freeze(log, inputs(log.claims))
    log.add(cb.label(VIEW, class_name='bus', class_id_hint=3, op_id='l2'))
    assert folded(inputs(log.claims)).state.class_name == 'bus'


def test_withdrawing_the_freeze_lets_the_undo_through_m1() -> None:
    log = Log()
    (label,) = log.add(cb.label(VIEW, class_name='truck', class_id_hint=2, op_id='l'))
    (freeze,) = _freeze(log, inputs(log.claims))
    log.add([cb.withdraw(label.claim_id, VIEW, op_id='u')])
    log.add(cb.forget_holdout_freeze(VIEW, [freeze.claim_id], op_id='x'))
    state = folded(
        inputs(log.claims, opinions=[opinion('classifier', 'car', outcome='model')])
    ).state
    assert (state.class_name, state.holdout) == ('car', False)


# ---- M17 ----------------------------------------------------------------------------------
def test_cleared_hides_only_older_m17() -> None:
    old = opinion('classifier', 'car', at=1, outcome='model')
    log = Log(clock=10)
    log.add(cb.clear_class(VIEW, op_id='c'))
    assert folded(inputs(log.claims, opinions=[old])).state.class_name is None
    newer = opinion('vlm', 'bus', at=100)
    assert folded(inputs(log.claims, opinions=[old, newer])).state.class_name == 'bus'


def test_an_empty_vlm_answer_keeps_the_class_m17() -> None:
    world = FoldWorld(None)
    world.apply(('vlm', 'matched', 'truck'))
    world.apply(('vlm', 'empty', 'car'))
    assert world.fold().state.class_name == 'truck'


def test_an_unresolved_class_keeps_the_lock_m17() -> None:
    log = Log()
    log.add(cb.label(VIEW, class_name='gone', class_id_hint=9, op_id='l'))
    state = folded(
        inputs(log.claims, opinions=[opinion('classifier', 'car', outcome='model')])
    ).state
    assert state.class_unresolved is True
    assert state.class_id is None
    assert state.class_validated is False
    assert state.label_locked is True  # fail closed: no machine class takes its place
    assert state.class_source == 'human'


# ---- M20 / M21 ----------------------------------------------------------------------------
def test_class_resolution_follows_merge_and_rename_m20() -> None:
    log = Log()
    log.add(cb.label(VIEW, class_name='car', class_id_hint=1, op_id='l'))
    merged = RegistrySnapshot(
        (RegistryClass(1, 'car', active=False, merged_into=2), RegistryClass(2, 'auto'))
    )
    assert folded(inputs(log.claims), merged).state.class_id == 2
    renamed = RegistrySnapshot((RegistryClass(1, 'lorry'), RegistryClass(5, 'car')))
    assert folded(inputs(log.claims), renamed).state.class_id == 5


def test_old_move_not_placed_m21() -> None:
    log = Log()
    log.add(cb.move_to_cluster(VIEW, partition_id='p1', cluster_uid='u1', op_id='m'))
    state = folded(inputs(log.claims)).state
    assert moved_placement(state, 'p1') == ('p1', 'u1')
    assert moved_placement(state, 'p2') is None  # a new partition serves: membership decides


# ---- order independence and rebuild --------------------------------------------------------
@pytest.mark.parametrize('seed', range(120))
def test_machine_results_order_independence(seed: int) -> None:
    rng = random.Random(seed)
    world = human_world(seed, length=5)
    machine_writes(world, seed, count=5)
    base = world.inputs()
    expect = fold_item(base, REGISTRY, SETTINGS)
    for _ in range(4):
        shuffled = dataclasses.replace(
            base,
            detections=tuple(rng.sample(base.detections, len(base.detections))),
            opinions=tuple(rng.sample(base.opinions, len(base.opinions))),
            boxes=tuple(rng.sample(base.boxes, len(base.boxes))),
            claims=tuple(rng.sample(base.claims, len(base.claims))),
        )
        assert fold_item(shuffled, REGISTRY, SETTINGS) == expect


@pytest.mark.parametrize('seed', range(120))
def test_fold_of_a_rebuilt_result_set_equals_the_incremental_one(seed: int) -> None:
    world = human_world(seed, length=5)
    machine_writes(world, seed, count=5)
    if any(c.kind is ClaimKind.CLASS_CLEARED for c in world.log.claims):
        pytest.skip('see test_rerun_after_a_clear_shows_older_results_again')
    incremental = world.fold()
    # empty every rebuildable result and run the analysers again, in another order and later
    rebuilt = dataclasses.replace(
        world.inputs(),
        detections=tuple(
            dataclasses.replace(d, created_at=d.created_at + timedelta(seconds=1000))
            for d in reversed(world.inputs().detections)
        ),
        opinions=tuple(
            dataclasses.replace(o, created_at=o.created_at + timedelta(seconds=1000))
            for o in reversed(world.inputs().opinions)
        ),
    )
    again = fold_item(rebuilt, REGISTRY, SETTINGS)
    assert again is not None
    if world.fold().state.holdout:
        pytest.skip('a frozen class is recorded in the claim, compared in the M1 tests')
    assert comparable(again) == comparable(incremental)


@pytest.mark.xfail(
    strict=True,
    reason='question 10: a clear hides what is older than itself, and a re-run makes every result '
    'newer, so plan 15 section 11 "empty every result table, re-run, state equals before" fails '
    'for an item a person cleared',
)
def test_rerun_after_a_clear_shows_older_results_again() -> None:
    old = opinion('classifier', 'car', at=1, outcome='model')
    log = Log(clock=10)
    log.add(cb.clear_class(VIEW, op_id='c'))
    before = folded(inputs(log.claims, opinions=[old])).state
    rerun = dataclasses.replace(old, created_at=t(500))
    after = folded(inputs(log.claims, opinions=[rerun])).state
    assert after.class_name == before.class_name


def test_fold_is_none_without_a_detection_or_an_asserting_claim() -> None:
    log = Log()
    log.add(cb.review_dismiss(VIEW))
    row = ItemRow(crop_id=CROP, photo_key=PHOTO, created_at=t(0))
    assert fold_item(inputs(log.claims, detections=[], item=row), REGISTRY, SETTINGS) is None
    log.add(cb.label(VIEW, class_name='car', class_id_hint=1, op_id='l'))
    created = fold_item(inputs(log.claims, detections=[], item=row), REGISTRY, SETTINGS)
    assert created is not None
    assert created.state.bbox == BOX  # an item made from its claims
