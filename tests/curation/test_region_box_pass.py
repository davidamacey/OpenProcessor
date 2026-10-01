"""W5: the ONE function that turns a detection pass's boxes into the stored
box-list write, shared by the worker's bulk writer and the test-on-crop
preview, so a preview is the write the worker would land."""

from __future__ import annotations

from typing import Any

import pytest

from src.config import get_region_fields
from src.config.region_state import RegionStatus
from src.services.curation.region_box_pass import box_pass_update, worker_stamps
from src.services.curation.region_boxes import RegionBox, new_box_placeholder


F = get_region_fields()


def _box(box_id: str, state: str = 'accepted', source: str = 'sam3', x: float = 0.1) -> RegionBox:
    return RegionBox(
        box_id=box_id, bbox_norm=(x, 0.1, x + 0.2, 0.3), state=state, source=source, score=0.9
    )


def _current(*boxes: RegionBox, seq: int = 0, revision: int = 4) -> dict:
    return {F.boxes: [b.to_doc() for b in boxes], F.box_seq: seq, F.revision: revision}


def test_fresh_boxes_get_real_ids_after_the_stored_high_water_mark() -> None:
    result = box_pass_update(
        _current(seq=7),
        [_box(new_box_placeholder(0)), _box(new_box_placeholder(1), x=0.5)],
        reverify=False,
        merge_machine_boxes=False,
        baseline=None,
        status=None,
        empty_status=RegionStatus.NO_REGION_BOX,
    )

    assert [b['box_id'] for b in result.update[F.boxes]] == ['b8', 'b9']
    assert result.update[F.status] == RegionStatus.DETECTED
    assert result.update[F.box_seq] == 9
    assert result.update[F.revision] == 5
    assert result.update[F.count] == 2


def test_a_fresh_detection_replaces_machine_boxes_and_keeps_human_ones() -> None:
    machine = _box('b1', source='sam3')
    human = _box('b2', source='human', x=0.5)

    result = box_pass_update(
        _current(machine, human, seq=2),
        [_box(new_box_placeholder(0), x=0.7)],
        reverify=False,
        merge_machine_boxes=True,
        baseline=None,
        status=None,
        empty_status=RegionStatus.NO_REGION_BOX,
    )

    ids = [b['box_id'] for b in result.update[F.boxes]]
    assert ids == ['b2', 'b3'], 'the stale machine box is gone, the human box survives'


def test_a_reverify_pass_resolves_its_box_in_place_and_carries_siblings() -> None:
    proposed = _box('b1', state='proposed')
    sibling = _box('b2', state='rejected', x=0.5)

    result = box_pass_update(
        _current(proposed, sibling, seq=2),
        [_box('b1', state='accepted')],
        reverify=True,
        merge_machine_boxes=False,
        baseline=[proposed, sibling],
        status=None,
        empty_status=RegionStatus.NO_REGION_BOX,
    )

    by_id = {b['box_id']: b['state'] for b in result.update[F.boxes]}
    assert by_id == {'b1': 'accepted', 'b2': 'rejected'}
    assert result.update[F.status] == RegionStatus.DETECTED


def test_no_boxes_keeps_the_empty_status_and_an_explicit_status_wins() -> None:
    empty = box_pass_update(
        _current(),
        [],
        reverify=False,
        merge_machine_boxes=False,
        baseline=None,
        status=None,
        empty_status=RegionStatus.NO_REGION_BOX,
    )
    forced = box_pass_update(
        _current(),
        [_box('b1', state='rejected')],
        reverify=False,
        merge_machine_boxes=False,
        baseline=None,
        status=RegionStatus.DETECTION_FAILED,
        empty_status=RegionStatus.NO_REGION_BOX,
    )

    assert empty.update[F.status] == RegionStatus.NO_REGION_BOX
    assert empty.update[F.boxes] == []
    assert forced.update[F.status] == RegionStatus.DETECTION_FAILED


def test_the_pass_does_not_touch_its_inputs() -> None:
    current = _current(_box('b1'), seq=1)
    before = {k: list(v) if isinstance(v, list) else v for k, v in current.items()}

    box_pass_update(
        current,
        [_box(new_box_placeholder(0), x=0.6)],
        reverify=False,
        merge_machine_boxes=True,
        baseline=None,
        status=None,
        empty_status=RegionStatus.NO_REGION_BOX,
    )

    assert current == before


@pytest.mark.parametrize('seq', [0, 3])
def test_merged_is_the_pre_finalization_list_for_the_embedding_entries(seq: int) -> None:
    result = box_pass_update(
        _current(seq=seq),
        [_box(new_box_placeholder(0))],
        reverify=False,
        merge_machine_boxes=False,
        baseline=None,
        status=None,
        empty_status=RegionStatus.NO_REGION_BOX,
    )

    assert result.merged[0].box_id == new_box_placeholder(0)
    assert result.finalized[0].box_id == f'b{seq + 1}'


def test_a_pass_with_no_status_at_all_is_refused() -> None:
    with pytest.raises(ValueError, match='status'):
        box_pass_update(
            _current(),
            [],
            reverify=False,
            merge_machine_boxes=False,
            baseline=None,
            status=None,
            empty_status=None,
        )


def _stamps(*, vlm_called: bool, profile_name: str | None = 'wheels') -> dict[str, Any]:
    return worker_stamps(
        profile_name=profile_name,
        profile_revision=3,
        pack_stamp='pack@2',
        vlm_called=vlm_called,
        vlm_endpoint='ep@1',
        vlm_model='real-model',
    )


def test_stamps_name_the_profile_and_only_claim_a_vlm_when_one_answered() -> None:
    assert _stamps(vlm_called=True) == {
        F.profile: 'wheels',
        F.profile_revision: 3,
        'vlm_prompt_pack': 'pack@2',
        'vlm_endpoint': 'ep@1',
        'vlm_model': 'real-model',
    }
    assert _stamps(vlm_called=False) == {F.profile: 'wheels', F.profile_revision: 3}
    assert F.profile not in _stamps(vlm_called=True, profile_name=None)


def _finalize(update: dict[str, Any], current: dict[str, Any], token: Any, **over: Any):
    from src.services.curation.region_box_pass import finalize_region_write

    kwargs: dict[str, Any] = {
        'doc_id': 'c1',
        'class_token': token,
        'trace': [],
        'stamps': {'region_profile': 'p'},
    }
    kwargs.update(over)
    return finalize_region_write(update, current, **kwargs)


def test_a_class_write_on_a_human_owned_class_is_stripped_but_the_region_write_lands() -> None:
    from src.services.curation.class_write_guard import class_state_token

    current = {'class_id': 1, 'class_name': 'a', 'class_source': 'human', 'class_validated': True}
    update = {'class_id': 2, 'class_name': 'b', 'class_source': 'vlm', F.status: 'detected'}

    out = _finalize(update, current, class_state_token(current))

    assert out[F.status] == 'detected'
    assert 'class_id' not in out
    assert 'class_id_history' not in out
    assert out['region_profile'] == 'p'


def test_an_update_stripped_to_nothing_stays_nothing_and_gets_no_stamps() -> None:
    from src.services.curation.class_write_guard import class_state_token

    current = {'class_id': 1, 'class_source': 'human', 'class_validated': True}

    assert (
        _finalize({'class_id': 2, 'class_source': 'vlm'}, current, class_state_token(current)) == {}
    )


def test_a_class_change_snapshots_the_previous_class_and_the_chain_merges() -> None:
    from src.services.curation.class_write_guard import class_state_token

    current = {
        'class_id': 1,
        'class_name': 'a',
        'class_source': 'item_model',
        F.detector_chain: ['x:hit'],
    }
    update = {'class_id': 2, 'class_name': 'b', 'class_source': 'vlm'}

    out = _finalize(update, current, class_state_token(current), trace=['sam3:hit'])

    assert out['class_id'] == 2
    assert out['class_id_history'][-1]['class_id'] == 1
    assert out[F.detector_chain] == ['x:hit', 'sam3:hit']
