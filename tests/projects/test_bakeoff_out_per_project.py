"""Bake-off output lives in each project's own dir, ``default`` included.

The router serves results from, and the pruner deletes under, one
per-project path. A flat ``<state_dir>/bakeoff_out`` shared by every
project would let one project's prune delete another's results.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest

from src.config.curation_accessors import base_curation_config
from src.config.project_context import bind_project
from src.config.projects import DEFAULT_SLUG, new_project_record


if TYPE_CHECKING:
    from pathlib import Path


def _record(slug: str, tmp_path: Path) -> Any:
    record = new_project_record(slug, base_curation_config())
    return replace(
        record,
        resources=replace(record.resources, bakeoff_results_dir=tmp_path / slug / 'bakeoff_jobs'),
    )


@pytest.mark.parametrize('slug', [DEFAULT_SLUG, 'alpha'])
def test_router_out_dir_is_the_projects_own(slug: str, tmp_path: Path) -> None:
    from src.routers.curation import bakeoff

    record = _record(slug, tmp_path)
    with bind_project(record):
        assert bakeoff._out_dir() == record.resources.bakeoff_results_dir / 'out'


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
    monkeypatch.setattr(prune_training_runs, 'list_runs', list)
    monkeypatch.setattr(prune_training_runs, 'plan_run_prune', lambda *_a, **_kw: [])
    monkeypatch.setattr(prune_training_runs, 'collect_active_bakeoff_run_ids', set)

    async def _no_promotions(_slug: str) -> list[Any]:
        return []

    monkeypatch.setattr(prune_training_runs.model_registry, 'owned_live', _no_promotions)
    record = _record(slug, tmp_path)
    args = argparse.Namespace(keep_last=3, bakeoff_out_keep_last=3, apply=False)
    with bind_project(record):
        prune_training_runs._run_one_project(args)
    assert seen == [record.resources.bakeoff_results_dir / 'out']


def test_export_prune_after_write_pins_the_projects_own_training_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A training run of the project pins its export: the post-export prune reads the bound
    project's ``training_runs`` rows, not a flat directory no run is ever written to."""
    from types import SimpleNamespace

    from src.services.curation import export_retention
    from src.services.curation.export_retention import prune_exports_after_write

    monkeypatch.setattr(export_retention, 'active_bakeoff_specs', list)

    names = [f'2026010{i}T000000Z' for i in range(1, 5)]
    for name in names:
        (tmp_path / 'exports' / name).mkdir(parents=True)
        (tmp_path / 'exports' / name / 'manifest.json').write_text('{}')
    oldest = tmp_path / 'exports' / names[0]
    monkeypatch.setattr(export_retention, '_run_export_dirs', lambda: {str(oldest)})
    config = SimpleNamespace(export_keep_last=1, export_root=tmp_path / 'exports')
    prune_exports_after_write(config)  # type: ignore[arg-type]
    assert oldest.is_dir(), 'pruned an export a training run still uses'
    assert not (tmp_path / 'exports' / names[1]).exists()


def test_the_results_directory_keeps_its_on_disk_path(tmp_path: Path) -> None:
    """The field was renamed from ``bakeoff_jobs_dir``; results already on disk must still be found."""
    from src.config.projects import resources_for_new

    base = base_curation_config()
    resources = resources_for_new('alpha', base)
    assert resources.bakeoff_results_dir == base.state_dir / 'projects' / 'alpha' / 'bakeoff_jobs'
    assert not hasattr(resources, 'bakeoff_jobs_dir')
    assert not hasattr(base, 'bakeoff_jobs_dir')
