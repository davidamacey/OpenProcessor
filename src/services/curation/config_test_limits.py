"""Bounds on the test-on-crop routes (W5, ``POST /prompt_packs/test`` and
``POST /region_profiles/test``).

A test run sends a real crop to a VLM or a segmenter, so each kind is capped
per process (the over-cap request is refused, not queued) and every run has
a wall-clock limit. Counters are plain integers: the event loop is single
threaded and nothing awaits between the check and the increment.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal


if TYPE_CHECKING:
    from collections.abc import Awaitable, Iterator

SlotKind = Literal['vlm', 'segmenter']

#: Concurrent test runs per process, per kind.
SLOT_CAPS: dict[str, int] = {'vlm': 2, 'segmenter': 4}
#: Most crop ids one pack-test request may name.
MAX_TEST_CROP_IDS = 64
#: Wall-clock limit of one test run, seconds.
TEST_TIMEOUT_S = 60.0

_active: dict[str, int] = {'vlm': 0, 'segmenter': 0}


class ConfigTestBusyError(RuntimeError):
    """Every slot of this kind is taken (``429 test_busy``)."""


class ConfigTestTimeoutError(RuntimeError):
    """The run exceeded :data:`TEST_TIMEOUT_S` (``504 test_timeout``)."""


@contextmanager
def reserve_slot(kind: SlotKind) -> Iterator[None]:
    """Hold one slot of ``kind`` for the body, or raise
    :class:`ConfigTestBusyError` at once when none is free."""
    if _active[kind] >= SLOT_CAPS[kind]:
        raise ConfigTestBusyError(kind)
    _active[kind] += 1
    try:
        yield
    finally:
        _active[kind] -= 1


async def bounded(awaitable: Awaitable[Any]) -> Any:
    """Await ``awaitable`` for at most :data:`TEST_TIMEOUT_S`."""
    try:
        return await asyncio.wait_for(awaitable, TEST_TIMEOUT_S)
    except TimeoutError as exc:
        raise ConfigTestTimeoutError from exc


def active_runs(kind: SlotKind) -> int:
    return _active[kind]


__all__ = [
    'SLOT_CAPS',
    'TEST_TIMEOUT_S',
    'ConfigTestBusyError',
    'ConfigTestTimeoutError',
    'active_runs',
    'bounded',
    'reserve_slot',
]
