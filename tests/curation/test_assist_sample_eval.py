"""``ask_sample``: the chunked VLM execution an assist run and the auto-label loop share.

The run's own behaviour is pinned by ``test_assist_run_classify.py`` and
``test_assist_run_lifecycle.py``; these tests pin what the extraction promises: chunking
by images per call, one outcome summed over the chunks, a caller-owned semaphore that bounds
in-flight chunks across concurrent callers, and a stop signal checked per chunk.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from src.services.curation.assist import sample_eval
from src.services.curation.assist.chunk import ChunkOutcome
from src.services.curation.assist.sample_eval import ask_sample
from src.services.curation.assist.slot_write import RunMeta


pytestmark = pytest.mark.asyncio

META = RunMeta(
    run_id='r1',
    created_at='2026-10-28T00:00:00+00:00',
    vlm_endpoint='ep',
    vlm_model='m',
    prompt_pack='pack',
    prompt_pack_revision=1,
)


class _Recorder:
    def __init__(self, delay: float = 0.0) -> None:
        self.calls: list[list[str]] = []
        self.in_flight = 0
        self.peak = 0
        self.delay = delay

    async def process_chunk(self, _os: Any, _ctx: Any, ids: list[str], **kw: Any) -> ChunkOutcome:
        assert kw['classify'] is True
        assert kw['attributes'] is False
        self.calls.append(list(ids))
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        await asyncio.sleep(self.delay)
        self.in_flight -= 1
        return ChunkOutcome(predicted=len(ids), updated=len(ids), failed=0)


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    rec = _Recorder()
    monkeypatch.setattr(sample_eval, 'process_chunk', rec.process_chunk)
    return rec


def _ask(ids: list[str], **kw: Any) -> Any:
    options: dict[str, Any] = {
        'actions': ['classify'],
        'box_ids': None,
        'images_per_call': 4,
        'meta': META,
        'semaphore': asyncio.Semaphore(4),
    }
    return ask_sample(None, None, ids, **{**options, **kw})  # type: ignore[arg-type]


async def test_chunks_by_images_per_call_and_sums_the_outcomes(recorder: _Recorder) -> None:
    ids = [f'c{i}' for i in range(10)]

    total = await _ask(ids, images_per_call=4)

    assert sorted(recorder.calls) == [
        ['c0', 'c1', 'c2', 'c3'],
        ['c4', 'c5', 'c6', 'c7'],
        ['c8', 'c9'],
    ]
    assert (total.predicted, total.updated) == (10, 10)


async def test_a_shared_semaphore_bounds_chunks_across_concurrent_callers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec = _Recorder(delay=0.01)
    monkeypatch.setattr(sample_eval, 'process_chunk', rec.process_chunk)
    shared = asyncio.Semaphore(3)
    ids_a = [f'a{i}' for i in range(20)]
    ids_b = [f'b{i}' for i in range(20)]

    await asyncio.gather(
        _ask(ids_a, images_per_call=1, semaphore=shared),
        _ask(ids_b, images_per_call=1, semaphore=shared),
    )

    assert len(rec.calls) == 40
    assert rec.peak == 3


async def test_a_stop_request_skips_the_chunks_not_yet_started(recorder: _Recorder) -> None:
    ids = [f'c{i}' for i in range(8)]
    asked: list[int] = []

    def stop() -> bool:
        return len(asked) >= 2

    total = await _ask(
        ids,
        images_per_call=2,
        semaphore=asyncio.Semaphore(1),
        should_stop=stop,
        on_chunk=asked.append,
    )

    assert len(recorder.calls) == 2
    assert total.predicted == 4


async def test_on_chunk_reports_each_chunk_size(recorder: _Recorder) -> None:
    sizes: list[int] = []

    await _ask([f'c{i}' for i in range(5)], images_per_call=2, on_chunk=sizes.append)

    assert sorted(sizes) == [1, 2, 2]


async def test_nothing_to_ask_calls_nothing(recorder: _Recorder) -> None:
    total = await _ask([])

    assert recorder.calls == []
    assert total == ChunkOutcome()
