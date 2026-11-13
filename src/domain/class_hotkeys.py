"""Which letter a new class gets as its one-key hotkey.

Pure: the registry passes in the letters already in use and the letters a keymap action owns, so
the choice is deterministic and can never collide with either."""

from __future__ import annotations

import string
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Collection

_LETTERS: str = string.ascii_lowercase


def pick_hotkey(name: str, taken: Collection[str], reserved: Collection[str]) -> str | None:
    """The first free letter: the name's own letters in order (its first letter first, so
    ``person`` gets ``p`` when it is free), then ``a`` to ``z``. ``None`` when every letter is
    taken or reserved; a class then has no letter until a person sets one."""
    blocked = {c.lower() for c in taken} | {c.lower() for c in reserved}
    own: list[str] = [c for c in name.lower() if c in _LETTERS]
    candidates: list[str] = [*own, *_LETTERS]
    for letter in candidates:
        if letter not in blocked:
            return letter
    return None


def free_letters(taken: Collection[str], reserved: Collection[str]) -> list[str]:
    """The letters ``a`` to ``z`` that no class holds and no keymap action owns."""
    blocked = {c.lower() for c in taken} | {c.lower() for c in reserved}
    return [c for c in _LETTERS if c not in blocked]
