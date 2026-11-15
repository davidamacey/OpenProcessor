"""The claim model: kinds, typed payloads, subject addressing, drafts (15a section 3)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.domain.claim_kinds import KINDS, ActorKind, ClaimKind, SubjectKind
from src.domain.claim_payloads import (
    PAYLOAD_MODELS,
    BoxTextSet,
    ClassSet,
    HoldoutFrozen,
    PayloadError,
    parse_payload,
    payload_json,
)
from src.domain.claims import Claim, ClaimDraft, ClaimError, Subject, commit


def t(n: float) -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=n)


BOX = (0.10, 0.10, 0.50, 0.50)
SUBJECT = Subject(SubjectKind.ITEM, photo_key=7, crop_id='c1', bbox=BOX, frame=1)


def _class_set(**over: object) -> dict[str, object]:
    base = {
        'class_name': 'car',
        'class_id_hint': 1,
        'label_source': 'human',
        'via': 'single',
        'labeler': 'human',
    }
    return {**base, **over}


def test_every_kind_has_a_payload_and_a_catalog_row() -> None:
    assert set(KINDS) == set(ClaimKind) == set(PAYLOAD_MODELS)


def test_catalog_flags_follow_the_plan_table() -> None:
    pins = {k for k, i in KINDS.items() if i.pins_geometry}
    asserts = {k for k, i in KINDS.items() if i.asserts_object}
    assert pins == {
        ClaimKind.ITEM_DRAWN,
        ClaimKind.GEOMETRY_SET,
        ClaimKind.CLASS_SET,
        ClaimKind.HOLDOUT_FROZEN,
        ClaimKind.EXCLUDED,  # a person's delete keeps the box (is_human_deleted)
        ClaimKind.REGION_STATUS_SET,
        ClaimKind.BOX_CREATED,
        ClaimKind.BOX_MOVED,
        ClaimKind.BOX_STATE_SET,
        ClaimKind.BOX_TEXT_SET,
        ClaimKind.BOX_DELETED,
    }
    assert ClaimKind.CLASS_CLEARED not in pins  # a clear, a move, dismissals and flags never pin
    assert ClaimKind.CLASS_MOVED not in pins
    assert asserts == {
        ClaimKind.ITEM_DRAWN,
        ClaimKind.GEOMETRY_SET,
        ClaimKind.CLASS_SET,
        ClaimKind.HOLDOUT_FROZEN,
        ClaimKind.BOX_CREATED,
        ClaimKind.BOX_MOVED,
        ClaimKind.BOX_STATE_SET,
        ClaimKind.BOX_TEXT_SET,
    }
    assert KINDS[ClaimKind.BOX_DELETED].asserts_object is False  # never creates a box


def test_class_set_roundtrips_through_json() -> None:
    payload = parse_payload(ClaimKind.CLASS_SET, _class_set())
    assert parse_payload(ClaimKind.CLASS_SET, payload_json(payload)) == payload


@pytest.mark.parametrize(
    ('kind', 'data'),
    [
        (ClaimKind.CLASS_SET, _class_set(label_source='robot')),
        (ClaimKind.CLASS_SET, _class_set(class_name='')),
        (ClaimKind.CLASS_SET, _class_set(via='telepathy')),
        (ClaimKind.CLASS_SET, _class_set(extra_field=1)),
        (ClaimKind.GEOMETRY_SET, {'bbox': (0.5, 0.1, 0.2, 0.9)}),
        (ClaimKind.GEOMETRY_SET, {'bbox': (0.1, 0.1, 1.2, 0.9)}),
        (ClaimKind.BOX_STATE_SET, {'state': 'maybe', 'region_seq': 1, 'item_bbox': BOX}),
        (ClaimKind.BOX_STATE_SET, {'state': 'accepted', 'item_bbox': BOX}),
        (ClaimKind.BOX_STATE_SET, {'state': 'accepted', 'region_seq': 1}),  # no item box seen
        (ClaimKind.CLASS_CLEARED, {'hidden': [{'source': 'vlm'}]}),
        (ClaimKind.HOLDOUT_FROZEN, {'freeze_id': 'f1'}),
        (
            ClaimKind.REGION_STATUS_SET,
            {'status': 'detected', 'label_source': 'human', 'region_seq': 0},
        ),
        (ClaimKind.EXCLUDED, {}),
        (ClaimKind.WITHDRAW, {'why': 'because'}),
    ],
)
def test_invalid_payloads_are_refused(kind: ClaimKind, data: dict[str, object]) -> None:
    with pytest.raises(PayloadError):
        parse_payload(kind, data)


def test_holdout_freeze_requires_the_machine_class_key_but_accepts_null() -> None:
    frozen_as_none = parse_payload(
        ClaimKind.HOLDOUT_FROZEN, {'freeze_id': 'f1', 'machine_class': None}
    )
    assert isinstance(frozen_as_none, HoldoutFrozen)
    assert frozen_as_none.machine_class is None  # M1: frozen as "no class" is a decision


def test_an_explicit_empty_box_text_is_a_reading() -> None:
    payload = parse_payload(ClaimKind.BOX_TEXT_SET, {'text': '', 'region_seq': 0, 'item_bbox': BOX})
    assert isinstance(payload, BoxTextSet)
    assert payload.text == ''


def test_payloads_are_frozen() -> None:
    payload = parse_payload(ClaimKind.CLASS_SET, _class_set())
    with pytest.raises(Exception, match='frozen'):
        payload.class_name = 'bus'  # type: ignore[attr-defined]


def test_item_and_box_subjects_need_what_the_person_saw() -> None:
    with pytest.raises(ClaimError, match='needs'):
        Subject(SubjectKind.ITEM, photo_key=1, crop_id='c')  # no bbox, no frame
    with pytest.raises(ClaimError, match='box_id'):
        Subject(SubjectKind.BOX, photo_key=1, crop_id='c', bbox=BOX, frame=1)
    with pytest.raises(ClaimError, match='both members'):
        Subject(SubjectKind.PAIR, crop_id='a')


def test_a_claim_must_match_its_kind() -> None:
    payload = ClassSet(class_name='car', label_source='human', via='single', labeler='human')
    with pytest.raises(ClaimError, match='about a item'):
        Claim(
            1,
            t(0),
            ActorKind.HUMAN,
            'api',
            None,
            ClaimKind.CLASS_SET,
            Subject(SubjectKind.CLAIM),
            payload,
        )
    with pytest.raises(ClaimError, match='payload'):
        Claim(1, t(0), ActorKind.HUMAN, 'api', None, ClaimKind.CLASS_CLEARED, SUBJECT, payload)
    with pytest.raises(ClaimError, match='withdraw'):
        ClaimDraft(ActorKind.HUMAN, 'api', None, ClaimKind.CLASS_SET, SUBJECT, payload, 5)


def test_a_withdraw_names_its_target_and_ranks_by_actor_then_id() -> None:
    withdraw = ClaimDraft(
        ActorKind.HUMAN,
        'api',
        'op',
        ClaimKind.WITHDRAW,
        Subject(SubjectKind.CLAIM),
        parse_payload(ClaimKind.WITHDRAW, {}),
        withdraws_claim_id=3,
    )
    row = commit(withdraw, claim_id=9, at=t(1))
    assert row.withdraws_claim_id == 3
    human = commit(
        ClaimDraft(
            ActorKind.HUMAN,
            'api',
            None,
            ClaimKind.RESTORED,
            SUBJECT,
            parse_payload(ClaimKind.RESTORED, {}),
        ),
        claim_id=1,
        at=t(0),
    )
    imported = commit(
        ClaimDraft(
            ActorKind.IMPORT,
            'import:i1',
            None,
            ClaimKind.RESTORED,
            SUBJECT,
            parse_payload(ClaimKind.RESTORED, {}),
        ),
        claim_id=2,
        at=t(0),
    )
    assert human.order > imported.order  # a person before an import, whatever the time
    assert imported.import_id == 'i1'
