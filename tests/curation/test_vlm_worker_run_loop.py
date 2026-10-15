"""``vlm_worker.run`` producer/consumer loop, pinned with fakes for the fetch, the
``label_batch`` call and the project discovery: chunking, in-flight exclusion, idle
stop and backoff, pause sentinel, error isolation, the console output and the
heartbeat. (The multi-project isolation proof is ``tests/projects/test_worker_leak.py``.)"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import httpx
import pytest

from scripts.curation import vlm_cli, vlm_worker


if TYPE_CHECKING:
    from pathlib import Path


def _record(slug: str) -> Any:
    return SimpleNamespace(slug=slug)


class Fakes:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.projects: list[Any] = [_record('alpha')]
        self.pending: dict[str, list[list[str]]] = {}
        self.fetch_calls: list[tuple[str, int, list[str]]] = []
        self.labelled: list[tuple[str, list[str]]] = []
        self.label_script: Any = None
        self.sleeps: list[float] = []
        self.heartbeats: list[tuple[str, dict[str, bool]]] = []
        self.closed = False
        self.label_gate: asyncio.Event | None = None
        self.fetch_gate: asyncio.Event | None = None
        self.bound: list[str] = []
        self.hooks: list[Any] = []
        m = monkeypatch
        m.setattr(vlm_worker, 'curation_api_prefix', lambda: '/curation')
        m.setattr(vlm_worker, 'script_project_registry', lambda _url: object())
        m.setattr(vlm_worker, 'make_script_opensearch', lambda *_a, **_kw: self)
        m.setattr(vlm_worker, 'projects_with_active_vlm', self._projects)
        m.setattr(vlm_worker, 'fetch_pending_ids', self._fetch)
        m.setattr(vlm_worker, 'label_batch', self._label)
        m.setattr(vlm_worker, 'heartbeat_loop', self._heartbeat)
        m.setattr(vlm_worker, 'bind_project', self._bind)
        real_sleep = asyncio.sleep

        async def _sleep(seconds: float) -> None:
            # The 30 s metrics reporter ticks are not what these tests count.
            if seconds != 30.0:
                self.sleeps.append(seconds)
            for hook in self.hooks:
                hook(seconds)
            await real_sleep(0)

        m.setattr(vlm_worker.asyncio, 'sleep', _sleep)

    async def close(self) -> None:
        self.closed = True

    async def _projects(self, _opensearch: Any, _registry: Any, only: str | None) -> list[Any]:
        return [p for p in self.projects if only in (None, p.slug)]

    async def _fetch(self, _opensearch: Any, **kw: Any) -> list[str]:
        if self.fetch_gate is not None:
            await self.fetch_gate.wait()
        slug = kw['slug']
        self.fetch_calls.append((slug, kw['batch_size'], list(kw['exclude_ids'])))
        queue = self.pending.get(slug, [])
        if isinstance(queue, Exception):
            raise queue
        return queue.pop(0) if queue else []

    async def _label(self, _client: Any, *, api: str, api_prefix: str, slug: str, crop_ids: Any):
        assert (api, api_prefix) == ('http://api.test', '/curation')
        if self.label_gate is not None:
            await self.label_gate.wait()
        self.labelled.append((slug, list(crop_ids)))
        if self.label_script is None:
            return {'predicted': len(crop_ids), 'updated': len(crop_ids) - 1}
        return self.label_script(slug, crop_ids)

    async def _heartbeat(self, name: str, status: Any, _stop: Any) -> None:
        self.heartbeats.append((name, status()))
        if self.fetch_gate is not None:
            self.fetch_gate.set()

    @contextlib.contextmanager
    def _bind(self, record: Any):
        self.bound.append(record.slug)
        yield


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Fakes:
    monkeypatch.setattr(vlm_worker.os, '_exit', lambda code: pytest.fail(f'os._exit({code})'))
    return Fakes(monkeypatch)


def _args(*extra: str) -> Any:
    return vlm_cli.parse_args(
        [
            '--api',
            'http://api.test',
            '--opensearch',
            'http://os.test',
            '--vlm-batch-size',
            '2',
            '--concurrency',
            '2',
            '--poll-interval',
            '0.5',
            '--pause-sentinel',
            '',
            *extra,
        ]
    )


@pytest.mark.asyncio
async def test_drain_pass_chunks_the_fetch_and_reports_the_session(
    fakes: Fakes, capsys: pytest.CaptureFixture[str]
) -> None:
    fakes.pending['alpha'] = [['c1', 'c2', 'c3', 'c4', 'c5']]
    assert await asyncio.wait_for(vlm_worker.run(_args('--until-empty')), 10) == 0

    assert sorted(fakes.labelled) == [
        ('alpha', ['c1', 'c2']),
        ('alpha', ['c3', 'c4']),
        ('alpha', ['c5']),
    ]
    # The fetch shares vlm_batch_size * concurrency * 2 over the one project.
    assert fakes.fetch_calls[0] == ('alpha', 8, [])
    assert fakes.closed is True
    assert fakes.bound == ['alpha', 'alpha']
    out = capsys.readouterr().out
    assert (
        '[vlm-worker] streaming: api=http://api.test vlm_batch=2 concurrency=2 queue_max=4 idle_stop_after=6'
        in out
    )
    assert '[vlm-worker] producer idle (1 empty polls) — draining' in out
    # Three chunks with 5 predicted and 2 updated (each chunk reports len-1).
    assert 'done: processed=5 updated=2 chunks=3 elapsed=' in out
    assert out.rstrip().endswith('cps')


@pytest.mark.asyncio
async def test_in_flight_ids_are_excluded_from_the_next_fetch_and_released(
    fakes: Fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    fakes.pending['alpha'] = [['c1', 'c2'], []]
    fakes.label_gate = asyncio.Event()
    original_fetch = fakes._fetch

    async def _fetch(opensearch: Any, **kw: Any) -> list[str]:
        ids = await original_fetch(opensearch, **kw)
        if len(fakes.fetch_calls) >= 2:
            fakes.label_gate.set()
        return ids

    monkeypatch.setattr(vlm_worker, 'fetch_pending_ids', _fetch)
    await asyncio.wait_for(vlm_worker.run(_args('--until-empty')), 10)

    assert fakes.fetch_calls[0][2] == []
    assert sorted(fakes.fetch_calls[1][2]) == ['c1', 'c2']


@pytest.mark.asyncio
async def test_fetch_is_split_evenly_over_the_active_projects(fakes: Fakes) -> None:
    fakes.projects = [_record('alpha'), _record('beta')]
    fakes.pending = {'alpha': [['a1']], 'beta': [['b1']]}
    await asyncio.wait_for(vlm_worker.run(_args('--until-empty')), 10)
    assert {(s, n) for s, n, _ in fakes.fetch_calls[:2]} == {('alpha', 4), ('beta', 4)}
    assert sorted(fakes.labelled) == [('alpha', ['a1']), ('beta', ['b1'])]


@pytest.mark.asyncio
async def test_a_failed_project_fetch_is_logged_and_the_others_continue(
    fakes: Fakes, capsys: pytest.CaptureFixture[str]
) -> None:
    fakes.projects = [_record('alpha'), _record('beta')]
    fakes.pending = {'alpha': RuntimeError('os down'), 'beta': [['b1']]}  # type: ignore[dict-item]
    await asyncio.wait_for(vlm_worker.run(_args('--until-empty')), 10)
    assert fakes.labelled == [('beta', ['b1'])]
    assert '[vlm-worker] project=alpha fetch error: os down' in capsys.readouterr().out


@pytest.mark.asyncio
async def test_no_classes_idles_once_and_does_not_count_the_chunk(
    fakes: Fakes, capsys: pytest.CaptureFixture[str]
) -> None:
    fakes.pending['alpha'] = [['c1', 'c2', 'c3', 'c4']]
    fakes.label_script = lambda _s, _ids: {'no_classes': True}
    await asyncio.wait_for(vlm_worker.run(_args('--until-empty')), 10)
    out = capsys.readouterr().out
    assert out.count('project has no classes yet; idling until some exist') == 1
    assert 'processed=0 updated=0 chunks=0' in out
    assert 0.5 in fakes.sleeps


@pytest.mark.asyncio
async def test_consumer_errors_are_logged_and_the_chunk_is_released(
    fakes: Fakes, capsys: pytest.CaptureFixture[str]
) -> None:
    fakes.pending['alpha'] = [['c1', 'c2', 'c3', 'c4', 'c5', 'c6']]

    def _script(_slug: str, ids: list[str]) -> dict[str, int]:
        if ids == ['c1', 'c2']:
            raise httpx.ConnectError('refused')
        if ids == ['c3', 'c4']:
            msg = 'bad json'
            raise ValueError(msg)
        return {'predicted': 2, 'updated': 2}

    fakes.label_script = _script
    await asyncio.wait_for(vlm_worker.run(_args('--until-empty')), 10)
    out = capsys.readouterr().out
    assert 'HTTP error: refused' in out
    assert 'error: bad json' in out
    assert 'processed=2 updated=2 chunks=1' in out


@pytest.mark.asyncio
async def test_progress_line_is_logged_every_tenth_chunk(
    fakes: Fakes, capsys: pytest.CaptureFixture[str]
) -> None:
    fakes.pending['alpha'] = [[f'c{i}' for i in range(20)]]
    await asyncio.wait_for(vlm_worker.run(_args('--until-empty', '--vlm-batch-size', '1')), 10)
    out = capsys.readouterr().out
    assert out.count('[vlm-worker] chunk ') == 2
    assert '[vlm-worker] chunk 10 (consumer ' in out
    assert '[vlm-worker] chunk 20 (consumer ' in out
    assert 'session=' in out


@pytest.mark.asyncio
async def test_idle_stop_after_n_empty_polls(
    fakes: Fakes, capsys: pytest.CaptureFixture[str]
) -> None:
    await asyncio.wait_for(vlm_worker.run(_args('--idle-stop-after', '3')), 10)
    assert len(fakes.fetch_calls) == 3
    assert 'producer idle (3 empty polls) — draining' in capsys.readouterr().out
    # Empty polls with a project present sleep the plain poll interval.
    assert fakes.sleeps.count(0.5) == 2


@pytest.mark.asyncio
async def test_no_active_project_backs_off_exponentially(fakes: Fakes) -> None:
    fakes.projects = []
    await asyncio.wait_for(vlm_worker.run(_args('--idle-stop-after', '4')), 10)
    assert fakes.fetch_calls == []
    assert [s for s in fakes.sleeps if s >= 0.5] == [0.5, 1.0, 2.0]


@pytest.mark.asyncio
async def test_continuous_mode_never_stops_on_idle(fakes: Fakes) -> None:
    def _stop_after_polls(_seconds: float) -> None:
        if len(fakes.sleeps) >= 8:
            os.kill(os.getpid(), signal.SIGINT)

    fakes.hooks.append(_stop_after_polls)
    await asyncio.wait_for(vlm_worker.run(_args('--continuous', '--idle-stop-after', '2')), 10)
    assert len(fakes.fetch_calls) >= 8


@pytest.mark.asyncio
async def test_pause_sentinel_holds_the_producer_until_removed(
    fakes: Fakes, tmp_path: Path
) -> None:
    sentinel = tmp_path / 'pause.sentinel'
    sentinel.write_text('1')
    fakes.pending['alpha'] = [['c1']]

    def _unpause(seconds: float) -> None:
        if seconds == 7.0:
            sentinel.unlink()

    fakes.hooks.append(_unpause)
    args = _args('--until-empty', '--pause-sentinel', str(sentinel), '--pause-poll-interval', '7')
    await asyncio.wait_for(vlm_worker.run(args), 10)
    assert fakes.sleeps[0] == 7.0
    assert fakes.labelled == [('alpha', ['c1'])]


@pytest.mark.asyncio
async def test_only_project_restricts_discovery(fakes: Fakes) -> None:
    fakes.projects = [_record('alpha'), _record('beta')]
    fakes.pending = {'alpha': [['a1']], 'beta': [['b1']]}
    await asyncio.wait_for(vlm_worker.run(_args('--until-empty', '--project', 'beta')), 10)
    assert fakes.labelled == [('beta', ['b1'])]


@pytest.mark.asyncio
async def test_heartbeat_reports_liveness_of_the_producer_and_consumers(fakes: Fakes) -> None:
    fakes.fetch_gate = asyncio.Event()
    await asyncio.wait_for(vlm_worker.run(_args('--until-empty')), 10)
    ((name, status),) = fakes.heartbeats
    assert name == 'vlm_worker'
    assert status == {'producer': True, 'consumers': True}


@pytest.mark.asyncio
async def test_unhandled_task_exception_is_fatal(
    fakes: Fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    exits: list[int] = []

    def _exit(code: int) -> None:
        exits.append(code)
        os.kill(os.getpid(), signal.SIGINT)

    monkeypatch.setattr(vlm_worker.os, '_exit', _exit)
    fakes._projects = AsyncMock(side_effect=RuntimeError('registry gone'))  # type: ignore[method-assign]
    monkeypatch.setattr(vlm_worker, 'projects_with_active_vlm', fakes._projects)
    with pytest.raises(RuntimeError, match='registry gone'):
        await asyncio.wait_for(vlm_worker.run(_args('--continuous')), 10)
    assert exits == [1]


@pytest.mark.asyncio
async def test_a_crop_released_during_a_fetch_is_not_dispatched_again(
    fakes: Fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """label_batch writes without a refresh, so a fetch that began before a chunk
    was released may still return that chunk's ids: they must be dropped."""
    labelled_once = asyncio.Event()
    fakes.pending['alpha'] = [['c1', 'c2'], ['c1', 'c2']]
    original_fetch = fakes._fetch

    async def _fetch(opensearch: Any, **kw: Any) -> list[str]:
        if len(fakes.fetch_calls) == 1:  # the second fetch: wait for the release
            fakes.fetch_calls.append(('alpha', kw['batch_size'], list(kw['exclude_ids'])))
            await labelled_once.wait()
            await asyncio.sleep(0)
            return fakes.pending['alpha'].pop(0)
        return await original_fetch(opensearch, **kw)

    monkeypatch.setattr(vlm_worker, 'fetch_pending_ids', _fetch)
    fakes.label_script = lambda _s, ids: (
        labelled_once.set() or {'predicted': len(ids), 'updated': len(ids)}
    )
    await asyncio.wait_for(vlm_worker.run(_args('--until-empty')), 10)
    assert fakes.labelled == [('alpha', ['c1', 'c2'])]


@pytest.mark.asyncio
async def test_no_classes_is_announced_again_after_a_successful_chunk(
    fakes: Fakes, capsys: pytest.CaptureFixture[str]
) -> None:
    fakes.pending['alpha'] = [['c1', 'c2', 'c3', 'c4', 'c5', 'c6']]
    replies = iter([{'no_classes': True}, {'predicted': 2, 'updated': 2}, {'no_classes': True}])
    fakes.label_script = lambda _s, _ids: next(replies)
    await asyncio.wait_for(vlm_worker.run(_args('--until-empty', '--concurrency', '1')), 10)
    out = capsys.readouterr().out
    assert out.count('project has no classes yet; idling until some exist') == 2
