"""Auto-activation only when no anchored or validated member moved and identities are stable."""

from __future__ import annotations

import pytest

from src.services.clustering.activation import auto_activation_allowed


def _diff(**kw):
    return {'anchored_moves': 0, 'validated_moves': 0, 'stability': 1.0, **kw}


def test_stability_one_activates() -> None:
    assert auto_activation_allowed(_diff(), activation='auto_when_safe')


def test_a_run_that_moves_an_anchored_member_stays_ready() -> None:
    assert not auto_activation_allowed(_diff(anchored_moves=1), activation='auto_when_safe')


def test_a_run_that_moves_a_validated_member_stays_ready() -> None:
    assert not auto_activation_allowed(_diff(validated_moves=3), activation='auto_when_safe')


@pytest.mark.parametrize(('stability', 'allowed'), [(0.95, True), (0.9499, False)])
def test_the_stability_floor_is_inclusive(stability: float, allowed: bool) -> None:
    assert (
        auto_activation_allowed(_diff(stability=stability), activation='auto_when_safe') is allowed
    )


def test_a_profile_that_wants_a_person_never_auto_activates() -> None:
    assert not auto_activation_allowed(_diff(), activation='human')
    assert not auto_activation_allowed(None, activation='auto_when_safe')
