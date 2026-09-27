"""Detection worker multi-project fairness (projects_plan.md §5.1).

Drives ``fetch_pending_multi_project`` -- the producer's per-cycle
discovery + deficit round-robin step -- against three projects with
backlogs of 1000 / 10 / 0 pending items. The fake ``_fetch_pending``
behaves like the real query: items stay pending until "written", and
``exclude_ids`` hides in-flight ones.
"""

from __future__ import annotations

import itertools
import math
from datetime import UTC, datetime
from typing import Any

import pytest

from scripts.curation._project_worker_utils import PIPELINE_PAUSED_FLAG_NAME
from scripts.curation.worker import cascade, fairness
from scripts.curation.worker.fairness import FairnessScheduler, fetch_pending_multi_project
from scripts.curation.worker.state import _ItemTask
from src.config.curation import base_curation_config
from src.config.project_context import current_project
from src.config.projects import ProjectRecord, resources_for_new


pytestmark = pytest.mark.unbound

FETCH_N = 30
QUEUE_MAX = 90  # per-project cap ceil(90 / 3) = 30 = FETCH_N


def _record(tmp_path: Any, slug: str) -> ProjectRecord:
    now = datetime.now(UTC).isoformat()
    resources = resources_for_new(slug, base_curation_config())
    state_dir = tmp_path / slug
    state_dir.mkdir(parents=True, exist_ok=True)
    resources = resources.__class__(**{**resources.__dict__, 'project_state_dir': state_dir})
    return ProjectRecord(
        slug=slug,
        display_name=slug,
        description='',
        status='active',
        revision=1,
        created_at=now,
        updated_at=now,
        origin=None,
        resources=resources,
    )


class _Registry:
    def __init__(self, projects: list[ProjectRecord]) -> None:
        self._projects = projects

    def active_projects(self) -> list[ProjectRecord]:
        return list(self._projects)


class _Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


class _Backlog:
    """Pending items per project. A fetch returns the oldest pending ids
    not excluded; items leave the backlog only when :meth:`write` runs."""

    def __init__(self, sizes: dict[str, int]) -> None:
        self.pending = {slug: [f'{slug}-{i}' for i in range(n)] for slug, n in sizes.items()}
        self.fetches: list[tuple[str, int]] = []

    async def fetch(
        self, _opensearch: Any, *, batch_size: int, exclude_ids: Any = None, project: Any = None
    ) -> list[_ItemTask]:
        bound = current_project().record.slug
        assert bound == project.slug, f'fetch for {project.slug} ran while {bound} was bound'
        excluded = set(exclude_ids or [])
        ids = [c for c in self.pending[project.slug] if c not in excluded][:batch_size]
        self.fetches.append((project.slug, batch_size))
        return [
            _ItemTask(
                crop_id=cid,
                image_path='/x.jpg',
                item_bbox_norm=(0.0, 0.0, 1.0, 1.0),
                region_status='pending_detection',
                class_name='',
                project=project,
            )
            for cid in ids
        ]

    def write(self, tasks: list[_ItemTask]) -> None:
        for t in tasks:
            self.pending[t.project.slug].remove(t.crop_id)


@pytest.fixture
def projects(tmp_path: Any) -> list[ProjectRecord]:
    return [_record(tmp_path, 'heavy'), _record(tmp_path, 'light'), _record(tmp_path, 'idle')]


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    c = _Clock()
    monkeypatch.setattr(fairness.time, 'monotonic', c)
    return c


@pytest.fixture
def backlog(monkeypatch: pytest.MonkeyPatch) -> _Backlog:
    b = _Backlog({'heavy': 1000, 'light': 10, 'idle': 0})
    monkeypatch.setattr(cascade, '_fetch_pending', b.fetch)
    return b


async def _cycle(
    projects: list[ProjectRecord],
    scheduler: FairnessScheduler,
    *,
    in_flight: list[_ItemTask] | None = None,
) -> list[_ItemTask]:
    counts: dict[str, int] = {}
    for t in in_flight or []:
        counts[t.project.slug] = counts.get(t.project.slug, 0) + 1
    scheduler.set_in_flight(counts)
    return await fetch_pending_multi_project(
        object(),
        registry=_Registry(projects),
        scheduler=scheduler,
        fetch_n=FETCH_N,
        queue_max=QUEUE_MAX,
        exclude_ids=[t.crop_id for t in in_flight or []],
    )


def _by_project(tasks: list[_ItemTask]) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in tasks:
        out[t.project.slug] = out.get(t.project.slug, 0) + 1
    return out


@pytest.mark.asyncio
async def test_small_project_drains_within_two_cycles_and_heavy_gets_leftover(
    projects: list[ProjectRecord], backlog: _Backlog, clock: _Clock
) -> None:
    scheduler = FairnessScheduler()
    drained_by = None
    for cycle in range(2):
        tasks = await _cycle(projects, scheduler)
        assert len({t.crop_id for t in tasks}) == len(tasks), 'an item was queued twice'
        counts = _by_project(tasks)
        # Work conserving: the cycle budget is used in full while heavy
        # still has a backlog, and heavy takes what light/idle left.
        assert len(tasks) == FETCH_N
        assert counts['heavy'] == FETCH_N - counts.get('light', 0)
        backlog.write(tasks)
        clock.t += 1.0
        if not backlog.pending['light'] and drained_by is None:
            drained_by = cycle
    assert drained_by is not None, 'the 10-item project was starved'


@pytest.mark.asyncio
async def test_busy_project_is_polled_every_cycle(
    projects: list[ProjectRecord], backlog: _Backlog, clock: _Clock
) -> None:
    scheduler = FairnessScheduler()
    for _ in range(5):
        tasks = await _cycle(projects, scheduler)
        assert _by_project(tasks).get('heavy', 0) > 0, 'a project with work must stay due'
        backlog.write(tasks)
        clock.t += 0.1


@pytest.mark.asyncio
async def test_idle_project_backs_off_from_5s_to_60s(
    projects: list[ProjectRecord], backlog: _Backlog, clock: _Clock
) -> None:
    scheduler = FairnessScheduler()
    idle_polls: list[float] = []
    for _ in range(400):
        before = len([f for f in backlog.fetches if f[0] == 'idle'])
        backlog.write(await _cycle(projects, scheduler))
        if len([f for f in backlog.fetches if f[0] == 'idle']) > before:
            idle_polls.append(clock.t)
        clock.t += 1.0
    gaps = [b - a for a, b in itertools.pairwise(idle_polls)]
    assert gaps[:4] == [5.0, 10.0, 20.0, 40.0]
    assert set(gaps[4:]) == {60.0}, 'backoff must cap at 60s'


@pytest.mark.asyncio
async def test_in_flight_cap_limits_one_project(
    projects: list[ProjectRecord], backlog: _Backlog, clock: _Clock
) -> None:
    scheduler = FairnessScheduler()
    cap = math.ceil(QUEUE_MAX / len(projects))
    first = await _cycle(projects, scheduler)
    assert _by_project(first)['heavy'] <= cap
    clock.t += 1.0
    # heavy now has ``cap`` items still in flight (not yet written).
    heavy_full = [
        _ItemTask(
            crop_id=f'heavy-{i}',
            image_path='/x.jpg',
            item_bbox_norm=(0.0, 0.0, 1.0, 1.0),
            region_status='pending_detection',
            class_name='',
            project=projects[0],
        )
        for i in range(cap)
    ]
    tasks = await _cycle(projects, scheduler, in_flight=heavy_full)
    assert _by_project(tasks).get('heavy', 0) == 0, 'a project at its in-flight cap got more'


@pytest.mark.asyncio
async def test_paused_project_stops_while_others_continue(
    projects: list[ProjectRecord], backlog: _Backlog, clock: _Clock
) -> None:
    heavy = projects[0]
    (heavy.resources.project_state_dir / PIPELINE_PAUSED_FLAG_NAME).write_text('1')
    scheduler = FairnessScheduler()
    fetched: dict[str, int] = {}
    for _ in range(2):
        tasks = await _cycle(projects, scheduler)
        for slug, n in _by_project(tasks).items():
            fetched[slug] = fetched.get(slug, 0) + n
        backlog.write(tasks)
        clock.t += 1.0
    assert 'heavy' not in {slug for slug, _ in backlog.fetches}, 'paused project was queried'
    assert fetched.get('light') == 10
