"""The letter a new class is given: deterministic, never taken, never a keymap action's key."""

from __future__ import annotations

import string

from src.domain.class_hotkeys import pick_hotkey


def test_the_first_letter_of_the_name_wins_when_free() -> None:
    assert pick_hotkey('person', set(), set()) == 'p'
    assert pick_hotkey('Traffic_light', set(), set()) == 't'


def test_a_taken_letter_falls_to_the_next_letter_of_the_name() -> None:
    assert pick_hotkey('person', {'p'}, set()) == 'e'


def test_a_reserved_letter_is_never_given() -> None:
    assert pick_hotkey('label', set(), {'l'}) == 'a'


def test_names_with_no_free_own_letter_get_the_first_free_alphabet_letter() -> None:
    assert pick_hotkey('aab', {'a', 'b'}, {'c'}) == 'd'
    assert pick_hotkey('1234', set(), set()) == 'a'


def test_nothing_free_gives_none() -> None:
    assert (
        pick_hotkey('x', set(string.ascii_lowercase[:13]), set(string.ascii_lowercase[13:])) is None
    )


def test_the_same_inputs_always_give_the_same_letter() -> None:
    first = [pick_hotkey(n, {'c'}, {'z'}) for n in ('car', 'cat', 'cup')]
    assert first == [pick_hotkey(n, {'c'}, {'z'}) for n in ('car', 'cat', 'cup')]
    assert first == ['a', 'a', 'u']
