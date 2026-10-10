"""The per-process CPU pool and the in-flight gate that keep ingest off the event loop (#241)."""

from __future__ import annotations

import asyncio
import contextvars
import threading
import time

import numpy as np
import pytest

from src.utils.cpu_pool import InflightGate, cpu_executor, run_cpu


def test_run_cpu_runs_off_the_loop_thread() -> None:
    async def main() -> tuple[int, int]:
        return threading.get_ident(), await run_cpu(threading.get_ident)

    loop_thread, call_thread = asyncio.run(main())
    assert call_thread != loop_thread
    assert cpu_executor()._thread_name_prefix == 'op-cpu-'


def test_run_cpu_carries_the_callers_contextvars() -> None:
    bound = contextvars.ContextVar('bound', default='unset')

    async def main() -> str:
        bound.set('project-a')
        return await run_cpu(bound.get)

    assert asyncio.run(main()) == 'project-a'


def test_run_cpu_passes_keyword_arguments() -> None:
    async def main() -> np.ndarray:
        return await run_cpu(np.stack, [np.zeros(2), np.ones(2)], axis=1)

    assert asyncio.run(main()).shape == (2, 2)


def test_a_slow_cpu_call_does_not_stop_the_event_loop() -> None:
    async def main() -> float:
        ticks: list[float] = []

        async def heartbeat() -> None:
            for _ in range(10):
                ticks.append(time.perf_counter())
                await asyncio.sleep(0.01)

        beat = asyncio.create_task(heartbeat())
        await run_cpu(time.sleep, 0.2)
        await beat
        return max(b - a for a, b in zip(ticks, ticks[1:], strict=False))

    # Run inline (the old behaviour) the loop would stall for the whole 0.2 s.
    assert asyncio.run(main()) < 0.1


def test_embed_crops_preprocessing_does_not_block_the_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.clients.pe_encoder as pe_encoder

    def slow_preprocess(crops: list[np.ndarray]) -> np.ndarray:
        time.sleep(0.2)
        return np.zeros((len(crops), 3, 336, 336), dtype=np.float32)

    monkeypatch.setattr(pe_encoder, '_preprocess_crops', slow_preprocess)

    class _Result:
        def as_numpy(self, _name: str) -> np.ndarray:
            return np.ones((1, 1024), dtype=np.float32)

    class _Pool:
        async def infer(self, _model, _inputs, **_kw):
            return _Result()

    encoder = pe_encoder.PEEncoder(triton_pool=_Pool())  # type: ignore[arg-type]

    async def main() -> float:
        gaps: list[float] = []

        async def heartbeat() -> None:
            last = time.perf_counter()
            for _ in range(15):
                await asyncio.sleep(0.01)
                now = time.perf_counter()
                gaps.append(now - last)
                last = now

        beat = asyncio.create_task(heartbeat())
        await encoder.embed_crops([np.zeros((40, 30, 3), dtype=np.uint8)])
        await beat
        return max(gaps)

    assert asyncio.run(main()) < 0.1


def test_gate_admits_up_to_capacity_and_queues_the_rest_in_order() -> None:
    async def main() -> list[str]:
        gate = InflightGate(4)
        order: list[str] = []
        release = asyncio.Event()

        async def request(name: str, units: int) -> None:
            async with gate.hold(units):
                order.append(f'in:{name}')
                await release.wait()
            order.append(f'out:{name}')

        tasks = [
            asyncio.create_task(request('a', 3)),
            asyncio.create_task(request('b', 3)),  # does not fit beside a
            asyncio.create_task(request('c', 1)),  # fits, but must not jump the queue past b
        ]
        await asyncio.sleep(0.05)
        assert order == ['in:a']
        assert gate.queued == 2
        release.set()
        await asyncio.gather(*tasks)
        assert gate.queued == 0
        return order

    order = asyncio.run(main())
    assert order.index('in:b') > order.index('out:a')
    assert order.index('in:c') > order.index('in:b')


def test_gate_clamps_a_request_larger_than_the_whole_capacity() -> None:
    async def main() -> bool:
        gate = InflightGate(2)
        async with gate.hold(10):
            return True

    assert asyncio.run(main())


def test_gate_gives_units_back_when_a_waiter_is_cancelled() -> None:
    async def main() -> None:
        gate = InflightGate(2)
        release = asyncio.Event()

        async def holder() -> None:
            async with gate.hold(2):
                await release.wait()

        first = asyncio.create_task(holder())
        await asyncio.sleep(0.01)
        waiter = asyncio.create_task(holder())
        await asyncio.sleep(0.01)
        assert gate.queued == 1
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
        assert gate.queued == 0
        release.set()
        await first
        async with gate.hold(2):  # all units are free again
            pass

    asyncio.run(main())


def test_gate_rejects_a_capacity_below_one() -> None:
    with pytest.raises(ValueError, match='at least 1'):
        InflightGate(0)
