"""Detection worker multi-project fairness (projects_plan.md §5.1).

RED-FIRST evidence (pre-fix): before ``fairness.py``/the multi-project
``fetch_pending_multi_project`` existed, this module's imports raised
``ModuleNotFoundError: No module named 'scripts.curation.worker.fairness'``
and ``ImportError: cannot import name 'fetch_pending_multi_project'`` --
the worker had no concept of more than one project (its producer called
``_fetch_pending`` once, unscoped, per cycle). See the task report for
the captured failure text.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from scripts.curation.worker import cascade
from scripts.curation.worker.fairness import (
    FairnessScheduler,
    fetch_pending_multi_project,
    is_project_paused,
)
from scripts.curation.worker.state import _ItemTask
from src.config.curation import base_curation_config
from src.config.projects import ProjectRecord, resources_for_new


pytestmark = pytest.mark.unbound


def _record(tmp_path: Any, slug: str, status: str = 'active') -> ProjectRecord:
    now = datetime.now(UTC).isoformat()
    resources = resources_for_new(slug, base_curation_config())
    # Redirect project_state_dir under tmp_path so the pause-flag file
    # and any liveness file this test writes never touch real disk
    # paths outside the test sandbox.
    state_dir = tmp_path / slug
    state_dir.mkdir(parents=True, exist_ok=True)
    resources = resources.__class__(**{**resources.__dict__, 'project_state_dir': state_dir})
    return ProjectRecord(
        slug=slug,
        display_name=slug,
        description='',
        status=status,  # type: ignore[arg-type]
        revision=1,
        created_at=now,
        updated_at=now,
        origin=None,
        resources=resources,
    )


class _FakeRegistry:
    def __init__(self, projects: list[ProjectRecord]) -> None:
        self._projects = projects

    def active_projects(self) -> list[ProjectRecord]:
        return [p for p in self._projects if p.status == 'active']


class _FakeOpenSearch:
    pass


def _backlog_fetcher(backlogs: dict[str, int]):
    """Monkeypatch target for ``cascade._fetch_pending``: pops up to
    ``batch_size`` fake tasks off ``backlogs[current project]``."""

    async def _fetch(opensearch: Any, *, batch_size: int, exclude_ids=None, project=None):
        slug = project.slug if project is not None else 'default'
        available = backlogs.get(slug, 0)
        n = min(batch_size, available)
        backlogs[slug] = available - n
        return [
            _ItemTask(
                crop_id=f'{slug}-{i}',
                image_path='/x.jpg',
                item_bbox_norm=(0.0, 0.0, 1.0, 1.0),
                region_status='pending_detection',
                class_name='',
                project=project,
            )
            for i in range(n)
        ]

    return _fetch


@pytest.fixture
def three_projects(tmp_path: Any) -> list[ProjectRecord]:
    return [
        _record(tmp_path, 'heavy'),
        _record(tmp_path, 'light'),
        _record(tmp_path, 'idle'),
    ]


async def _run_cycles(
    monkeypatch: pytest.MonkeyPatch,
    projects: list[ProjectRecord],
    backlogs: dict[str, int],
    n_cycles: int,
    *,
    fetch_n: int = 30,
    queue_max: int = 12,
) -> list[dict[str, int]]:
    monkeypatch.setattr(cascade, '_fetch_pending', _backlog_fetcher(backlogs))
    registry = _FakeRegistry(projects)
    scheduler = FairnessScheduler()
    per_cycle: list[dict[str, int]] = []
    last_backlog: dict[str, int] = {}
    for _ in range(n_cycles):
        tasks = await fetch_pending_multi_project(
            _FakeOpenSearch(),
            registry=registry,
            scheduler=scheduler,
            fetch_n=fetch_n,
            queue_max=queue_max,
            backlog_hint=last_backlog,
        )
        counts: dict[str, int] = {}
        for t in tasks:
            counts[t.project.slug] = counts.get(t.project.slug, 0) + 1
        per_cycle.append(counts)
        last_backlog = dict(counts)
    return per_cycle


@pytest.mark.asyncio
async def test_light_project_drains_without_starvation(
    monkeypatch: pytest.MonkeyPatch, three_projects: list[ProjectRecord]
) -> None:
    backlogs = {'heavy': 1000, 'light': 10, 'idle': 0}
    per_cycle = await _run_cycles(monkeypatch, three_projects, backlogs, n_cycles=2)
    fetched_light = sum(c.get('light', 0) for c in per_cycle)
    assert fetched_light == 10, f'light project starved: fetched only {fetched_light}/10'


@pytest.mark.asyncio
async def test_heavy_project_gets_leftover_capacity(
    monkeypatch: pytest.MonkeyPatch, three_projects: list[ProjectRecord]
) -> None:
    backlogs = {'heavy': 1000, 'light': 10, 'idle': 0}
    per_cycle = await _run_cycles(monkeypatch, three_projects, backlogs, n_cycles=1, fetch_n=30)
    # heavy should get most of the 30-item cycle budget once light's
    # small ask and idle's zero ask are satisfied.
    assert per_cycle[0].get('heavy', 0) >= 15


@pytest.mark.asyncio
async def test_idle_project_backs_off(
    monkeypatch: pytest.MonkeyPatch, three_projects: list[ProjectRecord]
) -> None:
    backlogs = {'heavy': 1000, 'light': 10, 'idle': 0}
    monkeypatch.setattr(cascade, '_fetch_pending', _backlog_fetcher(backlogs))
    registry = _FakeRegistry(three_projects)
    scheduler = FairnessScheduler()
    intervals = []
    now = 1000.0
    last_backlog: dict[str, int] = {}
    for _ in range(5):
        tasks = await fetch_pending_multi_project(
            _FakeOpenSearch(),
            registry=registry,
            scheduler=scheduler,
            fetch_n=30,
            queue_max=12,
            backlog_hint=last_backlog,
        )
        last_backlog = {}
        for t in tasks:
            last_backlog[t.project.slug] = last_backlog.get(t.project.slug, 0) + 1
        intervals.append(scheduler.poll_interval('idle'))
        # Advance the scheduler's idea of "now" past this project's
        # next-due time so the next cycle actually re-polls it.
        now += scheduler.poll_interval('idle') + 0.01
        scheduler._poll_state['idle'].next_due_at = now - 0.01

    assert intervals == sorted(intervals), f'idle backoff never grew monotonically: {intervals}'
    assert intervals[-1] > intervals[0], 'idle project never backed off'
    assert intervals[-1] <= 60.0


@pytest.mark.asyncio
async def test_paused_project_is_skipped(
    monkeypatch: pytest.MonkeyPatch, three_projects: list[ProjectRecord]
) -> None:
    heavy, light, _idle = three_projects
    (heavy.resources.project_state_dir / 'pipeline_paused.flag').write_text('1')
    assert is_project_paused(heavy) is True
    assert is_project_paused(light) is False

    backlogs = {'heavy': 1000, 'light': 10, 'idle': 0}
    per_cycle = await _run_cycles(monkeypatch, three_projects, backlogs, n_cycles=2)
    fetched_heavy = sum(c.get('heavy', 0) for c in per_cycle)
    fetched_light = sum(c.get('light', 0) for c in per_cycle)
    assert fetched_heavy == 0, 'paused project must not be fetched'
    assert fetched_light == 10, 'unpaused projects must keep making progress while one is paused'
