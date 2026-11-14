"""The region's offset allocator: aligned, coalescing, and a bounded wait."""

from __future__ import annotations

import random
import threading
import time

import pytest

from frame_engine.slab import ALIGN, SlabAllocator, SlabTimeoutError


def _free(slab: SlabAllocator) -> list[tuple[int, int]]:
    return [(o, n) for o, n in slab._free]


def test_offsets_are_aligned_and_sizes_rounded_up() -> None:
    slab = SlabAllocator(16 * ALIGN)
    a = slab.alloc(1, 0)
    b = slab.alloc(ALIGN + 1, 0)
    assert (a, b) == (0, ALIGN)
    assert slab.in_use == 3 * ALIGN
    assert all(off % ALIGN == 0 for off, _ in _free(slab))


def test_released_neighbours_coalesce_into_one_block() -> None:
    slab = SlabAllocator(8 * ALIGN)
    offs = [slab.alloc(2 * ALIGN, 0) for _ in range(4)]
    for off in (offs[1], offs[3], offs[0], offs[2]):
        slab.release(off, 2 * ALIGN)
    assert _free(slab) == [(0, 8 * ALIGN)]
    assert slab.in_use == 0
    assert slab.peak == 8 * ALIGN
    assert slab.alloc(8 * ALIGN, 0) == 0  # the whole region again


def test_a_full_region_times_out_within_its_bound() -> None:
    slab = SlabAllocator(4 * ALIGN)
    slab.alloc(4 * ALIGN, 0)
    start = time.monotonic()
    with pytest.raises(SlabTimeoutError):
        slab.alloc(ALIGN, 0.1)
    assert 0.09 <= time.monotonic() - start < 2.0
    assert slab.waits == 1


def test_a_waiter_proceeds_when_room_is_released() -> None:
    slab = SlabAllocator(4 * ALIGN)
    first = slab.alloc(4 * ALIGN, 0)
    got: list[int] = []
    t = threading.Thread(target=lambda: got.append(slab.alloc(2 * ALIGN, 5)))
    t.start()
    time.sleep(0.05)
    slab.release(first, 4 * ALIGN)
    t.join(5)
    assert got == [0]


def test_a_request_larger_than_the_region_is_refused_at_once() -> None:
    slab = SlabAllocator(4 * ALIGN)
    with pytest.raises(ValueError, match='exceeds the region'):
        slab.alloc(4 * ALIGN + 1, 10)


def test_the_size_must_be_a_positive_multiple_of_the_alignment() -> None:
    with pytest.raises(ValueError, match='positive multiple'):
        SlabAllocator(ALIGN + 1)


def test_many_threads_never_overlap_and_return_everything() -> None:
    slab = SlabAllocator(64 * ALIGN)
    held: dict[int, int] = {}
    lock = threading.Lock()
    errors: list[str] = []

    def work(seed: int) -> None:
        rng = random.Random(seed)
        for _ in range(200):
            n = rng.randint(1, 8) * ALIGN - rng.randint(0, ALIGN - 1)
            off = slab.alloc(n, 5)
            with lock:
                for o, m in held.items():
                    if off < o + m and o < off + n:
                        errors.append(f'{off}+{n} overlaps {o}+{m}')
                held[off] = n
            with lock:
                del held[off]
            slab.release(off, n)

    threads = [threading.Thread(target=work, args=(k,)) for k in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert errors == []
    assert slab.in_use == 0
    assert _free(slab) == [(0, 64 * ALIGN)]
