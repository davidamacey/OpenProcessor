"""The device-memory budget of ``gpu_image_prep`` (``models/gpu_image_prep/1/prep_budget.py``)."""

from __future__ import annotations

import importlib
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest


MODEL_DIR = Path(__file__).resolve().parents[2] / 'models' / 'gpu_image_prep' / '1'


@pytest.fixture
def pb(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(MODEL_DIR))
    for name in ('prep_budget', 'prep_geometry'):
        monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module('prep_budget')


def _header(w: int, h: int, orientation: int = 1) -> Any:
    geo = importlib.import_module('prep_geometry')
    return geo.JpegHeader('jpeg', w, h, 3, 8, False, orientation, False)


def test_need_bytes_is_the_frame_for_an_upright_photo(pb: Any) -> None:
    assert pb.need_bytes(_header(5472, 3648)) == 5472 * 3648 * 3


@pytest.mark.parametrize('orientation', [2, 3, 4, 5, 6, 7, 8])
def test_need_bytes_adds_the_oriented_copy_when_the_tag_is_not_one(
    pb: Any, orientation: int
) -> None:
    assert pb.need_bytes(_header(4000, 3000, orientation)) == 2 * 4000 * 3000 * 3


def test_a_grey_photo_is_still_charged_the_three_channel_output(pb: Any) -> None:
    geo = importlib.import_module('prep_geometry')
    grey = geo.JpegHeader('jpeg', 100, 50, 1, 8, False, 1, False)
    assert pb.need_bytes(grey) == 100 * 50 * 3


def test_acquire_and_release_track_the_bytes_in_use(pb: Any) -> None:
    budget = pb.ByteBudget(100)
    assert budget.acquire(60, timeout=0.1)
    assert budget.in_use == 60
    assert not budget.acquire(60, timeout=0.05)  # waits, then gives up
    assert budget.in_use == 60
    budget.release(60)
    assert budget.in_use == 0
    assert budget.acquire(100, timeout=0.1)


def test_a_waiter_proceeds_when_bytes_are_released(pb: Any) -> None:
    budget = pb.ByteBudget(100)
    assert budget.acquire(80, timeout=0.1)
    got: list[bool] = []
    t = threading.Thread(target=lambda: got.append(budget.acquire(50, timeout=5)))
    t.start()
    time.sleep(0.05)
    assert got == []
    budget.release(80)
    t.join(5)
    assert got == [True]


def test_waiters_are_served_first_come_first_served(pb: Any) -> None:
    """A small request behind a large one must not overtake it (the large one would starve)."""
    budget = pb.ByteBudget(100)
    assert budget.acquire(90, timeout=0.1)
    order: list[str] = []

    def take(name: str, n: int) -> None:
        assert budget.acquire(n, timeout=5)
        order.append(name)
        budget.release(n)

    big = threading.Thread(target=take, args=('big', 80))
    big.start()
    time.sleep(0.05)
    small = threading.Thread(target=take, args=('small', 5))
    small.start()
    time.sleep(0.05)
    assert order == []  # 10 bytes are free, yet small waits behind big
    budget.release(90)
    big.join(5)
    small.join(5)
    assert order == ['big', 'small']


def test_a_timed_out_waiter_does_not_block_those_behind_it(pb: Any) -> None:
    budget = pb.ByteBudget(100)
    assert budget.acquire(90, timeout=0.1)
    got: list[bool] = []
    big = threading.Thread(target=lambda: got.append(budget.acquire(80, timeout=0.2)))
    big.start()
    time.sleep(0.05)
    small = threading.Thread(target=lambda: got.append(budget.acquire(5, timeout=5)))
    small.start()
    big.join(5)
    small.join(5)
    assert sorted(got) == [False, True]


def test_an_oversized_request_is_refused_not_queued(pb: Any) -> None:
    budget = pb.ByteBudget(100)
    with pytest.raises(pb.OverBudgetError):
        budget.acquire(101, timeout=1)


def test_sixty_four_threads_with_random_sizes_neither_deadlock_nor_overspend(pb: Any) -> None:
    budget = pb.ByteBudget(1000)
    peak = [0]
    lock = threading.Lock()
    errors: list[str] = []

    def work(seed: int) -> None:
        rng = random.Random(seed)
        for _ in range(30):
            n = rng.randint(1, 600)
            if not budget.acquire(n, timeout=30):
                errors.append('timeout')
                return
            with lock:
                peak[0] = max(peak[0], budget.in_use)
            time.sleep(rng.random() * 0.001)
            budget.release(n)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(64)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert not any(t.is_alive() for t in threads)
    assert errors == []
    assert peak[0] <= 1000
    assert budget.in_use == 0


def test_memory_errors_are_told_from_other_failures(pb: Any) -> None:
    class OutOfMemoryError(Exception):
        pass

    class CUDARuntimeError(Exception):
        def __init__(self, status: int) -> None:
            super().__init__(f'cudaError {status}')
            self.status = status

    class NvJpegError(RuntimeError):
        def __init__(self, status: int) -> None:
            super().__init__(f'nvjpegStatus {status}')
            self.status = status

    assert pb.is_memory_error(OutOfMemoryError('Out of memory allocating 63 MB'))
    assert pb.is_memory_error(CUDARuntimeError(2))  # cudaErrorMemoryAllocation
    assert not pb.is_memory_error(CUDARuntimeError(700))
    assert pb.is_memory_error(NvJpegError(5))  # NVJPEG_STATUS_ALLOCATOR_FAILURE
    assert not pb.is_memory_error(NvJpegError(2))
    assert not pb.is_memory_error(ValueError('bad'))
