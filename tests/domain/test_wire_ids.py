"""A wire id is 32 lowercase hex digits and is stored as a uuid; the round trip is the identity."""

from __future__ import annotations

import uuid

import pytest

from src.domain.wire_ids import WireIdError, to_uuid, to_wire


def test_a_wire_id_round_trips_through_its_stored_form() -> None:
    wire = 'a' * 8 + '0123' * 6
    assert to_wire(to_uuid(wire)) == wire
    assert to_uuid(wire) == uuid.UUID(wire)


@pytest.mark.parametrize(
    'bad', ['', 'A' * 32, 'a' * 31, 'a' * 33, 'g' * 32, str(uuid.uuid4()), ' ' + 'a' * 31]
)
def test_anything_else_is_refused_not_guessed_at(bad: str) -> None:
    with pytest.raises(WireIdError):
        to_uuid(bad)


def test_the_rendering_is_lowercase_without_hyphens() -> None:
    value = uuid.UUID('ABCDEF01-2345-6789-ABCD-EF0123456789')
    assert to_wire(value) == 'abcdef0123456789abcdef0123456789'
