"""Bake-off output lives in each project's own dir, ``default`` included.

The router serves results from, and the pruner deletes under, one
per-project path. A flat ``<state_dir>/bakeoff_out`` shared by every
project would let one project's prune delete another's results.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest

from src.config.curation import base_curation_config
from src.config.project_context import bind_project
from src.config.projects import DEFAULT_SLUG, new_project_record


if TYPE_CHECKING:
    from pathlib import Path


def _record(slug: str, tmp_path: Path) -> Any:
    record = new_project_record(slug, base_curation_config())
    return replace(
        record,
        resources=replace(record.resources, bakeoff_jobs_dir=tmp_path / slug / 'bakeoff_jobs'),
    )


@pytest.mark.parametrize('slug', [DEFAULT_SLUG, 'alpha'])
def test_router_out_dir_is_the_projects_own(slug: str, tmp_path: Path) -> None:
    from src.routers.curation import bakeoff

    record = _record(slug, tmp_path)
    with bind_project(record):
        assert bakeoff._out_dir() == record.resources.bakeoff_jobs_dir / 'out'


@pytest.mark.parametrize('slug', [DEFAULT_SLUG, 'alpha'])
def test_prune_plans_the_bound_projects_own_out_dir(
    slug: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import argparse

    from scripts.curation import prune_training_runs

    seen: list[Path] = []

    def _plan_out(out_dir: Path, _keep: int) -> list[Any]:
        seen.append(out_dir)
        return []

    monkeypatch.setattr(prune_training_runs, 'plan_bakeoff_out_prune', _plan_out)
    monkeypatch.setattr(prune_training_runs, 'plan_run_prune', lambda **_kw: [])
    monkeypatch.setattr(prune_training_runs, 'resolve_triton_models_dir', lambda: tmp_path)
    record = _record(slug, tmp_path)
    args = argparse.Namespace(keep_last=3, bakeoff_out_keep_last=3, apply=False)
    with bind_project(record):
        prune_training_runs._run_one_project(args)
    assert seen == [record.resources.bakeoff_jobs_dir / 'out']


def test_export_prune_after_write_pins_the_projects_own_training_job(tmp_path: Path) -> None:
    """A training job queued in the project's own ``train_jobs_dir`` pins
    its export: the post-export prune must read the project's job dirs,
    not a flat env root no job is ever written to."""
    import json
    from types import SimpleNamespace

    from src.config import get_curation_config
    from src.services.curation.export_retention import prune_exports_after_write

    record = new_project_record('alpha', base_curation_config())
    record = replace(
        record,
        resources=replace(
            record.resources,
            export_root=tmp_path / 'exports',
            train_jobs_dir=tmp_path / 'jobs' / 'projects' / 'alpha',
            bakeoff_jobs_dir=tmp_path / 'state' / 'projects' / 'alpha' / 'bakeoff_jobs',
        ),
    )
    names = [f'2026010{i}T000000Z' for i in range(1, 5)]
    for name in names:
        (tmp_path / 'exports' / name).mkdir(parents=True)
        (tmp_path / 'exports' / name / 'manifest.json').write_text('{}')
    oldest = tmp_path / 'exports' / names[0]
    record.resources.train_jobs_dir.mkdir(parents=True)
    (record.resources.train_jobs_dir / 'r1.job.json').write_text(
        json.dumps({'dataset_export_dir': str(oldest)})
    )
    with bind_project(record):
        view = get_curation_config()
        config = SimpleNamespace(
            export_keep_last=1,
            export_root=view.export_root,
            train_jobs_dir=view.train_jobs_dir,
            bakeoff_jobs_dir=view.bakeoff_jobs_dir,
        )
        prune_exports_after_write(config)  # type: ignore[arg-type]
    assert oldest.is_dir(), 'pruned an export a queued training job still uses'
    assert not (tmp_path / 'exports' / names[1]).exists()
