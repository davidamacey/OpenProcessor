"""Two projects' loops take turns by slice and never see each other's rows or items."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import psycopg
import pytest

from curation.assist_loop_fakes import ITEMS, FakeVlm, LoopEnv, decisions, drive, make_world


if TYPE_CHECKING:
    from pathlib import Path

    from fixtures.postgres import PgDatabase

pytestmark = [pytest.mark.db, pytest.mark.usefixtures('readable_crops')]


def test_two_projects_interleave_their_slices_and_end_with_their_own_labels(
    pg: PgDatabase,
    tmp_path: Path,
    captured_events: list[dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del captured_events
    first_world = make_world(monkeypatch)
    first_world.add_cluster(10001, {'car': 60, 'truck': 60, 'bus': 60, 'van': 60}, prefix='a')
    second_world = make_world(monkeypatch)
    second_world.add_cluster(20001, {'truck': 300}, prefix='b')
    envs = [LoopEnv.make(pg, tmp_path / 'one'), LoopEnv.make(pg, tmp_path / 'two')]
    worlds = [first_world, second_world]
    vlms = [FakeVlm(first_world.truth), FakeVlm(second_world.truth)]
    refs = [env.new_run() for env in envs]
    done = [False, False]
    turns: list[int] = []

    while not all(done):
        for index in (0, 1):
            if done[index]:
                continue
            start = vlms[index].requests
            run = asyncio.run(
                drive(
                    worlds[index],
                    envs[index],
                    vlms[index],
                    ref=refs[index],
                    node_concurrency=1,
                    slice_due=lambda i=index, s=start: vlms[i].requests > s,
                )
            )
            turns.append(index)
            done[index] = run.outcome.status == 'completed'
        assert len(turns) < 60

    assert set(turns[:2]) == {0, 1}, 'the second project did not wait for the first'
    assert {d['class_name'] for d in first_world.os.docs(ITEMS).values()} == {
        'car',
        'truck',
        'bus',
        'van',
    }
    assert {d['class_name'] for d in second_world.os.docs(ITEMS).values()} == {'truck'}
    with psycopg.connect(pg.conninfo('super'), autocommit=True) as conn:
        for env, ref in zip(envs, refs, strict=True):
            owner = conn.execute(
                'SELECT project_id FROM assist_loop_runs WHERE id = %s', (ref.run_id,)
            ).fetchone()[0]
            assert owner == env.project
    assert decisions(run.record)
    assert refs[0].run_id != refs[1].run_id
