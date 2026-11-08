"""The loop's durable record is its tables: a slice that dies leaves everything a next slice
needs in ``assist_loop_*``, nothing in ``jobs.params``, ``jobs.checkpoint`` or ``events``.
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from curation.assist_loop_fakes import ITEMS, FakeVlm, World, decisions, drive, make_world, reload
from src.domain.loop_stats import required_sample


if TYPE_CHECKING:
    from curation.assist_loop_fakes import LoopEnv


SRC = Path(__file__).resolve().parents[2] / 'src'
LOOP_MODULES = [
    *sorted((SRC / 'services/curation/assist_loop').glob('*.py')),
    SRC / 'tasks/assist_loop.py',
]
FORBIDDEN_ATTRS = {'checkpoint', 'resume_from'}


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> World:
    return make_world(monkeypatch)


@pytest.mark.db
@pytest.mark.usefixtures('readable_crops')
def test_a_passed_cluster_is_in_the_tables_and_a_fresh_slice_reads_it_back(
    world: World, loop_env: LoopEnv
) -> None:
    world.add_cluster(10001, {'car': 300}, prefix='a')

    run = asyncio.run(drive(world, loop_env, FakeVlm(world.truth)))

    k1 = required_sample(300, 0.9, 0.025, 0)
    assert run.outcome.status == 'completed'
    assert run.outcome.stop_reason == 'done'
    fresh = asyncio.run(reload(loop_env, run.ref))
    node = fresh.tree.node('r10001')
    assert node.step == 'done'
    assert decisions(fresh) == {'r10001': 'pass'}
    assert node.propagated is not None
    assert node.propagated['written'] == 300 - k1
    assert len(node.answers) == k1
    assert node.crops == k1
    cars = [d for d in world.os.docs(ITEMS).values() if d['class_name'] == 'car']
    assert len(cars) == 300


@pytest.mark.parametrize('module', LOOP_MODULES, ids=lambda p: p.name)
def test_no_loop_module_reads_the_job_checkpoint_or_the_events_feed(module: Path) -> None:
    tree = ast.parse(module.read_text(encoding='utf-8'))
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not names & FORBIDDEN_ATTRS, module.name
    imported = {
        a.name
        for n in ast.walk(tree)
        if isinstance(n, (ast.Import, ast.ImportFrom))
        for a in n.names
    }
    assert 'job_events' not in imported, module.name
