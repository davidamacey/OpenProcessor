#!/usr/bin/env python3
"""Keep-last retention for finished training runs + bake-off output (ST-4).

Two independent prunes, both dry-run by default:

* training-run job state under ``OP_TRAIN_JOBS_DIR`` -- never a
  non-terminal run, a promoted run, or one a queued/running bake-off job
  still references (see ``src.services.training.run_retention``);
* bake-off comparison output under ``OP_BAKEOFF_OUT_DIR`` -- pure
  keep-last, no pin logic (scored output, not a training artifact).

Neither touches MLflow runs.

Loops over every active AND archived project by default, pruning each
project's own runs under its own binding (projects_plan.md §11 W7).
``--project SLUG`` restricts the run to just that one project.

    # See what would be removed, every project.
    python3 scripts/curation/prune_training_runs.py

    # Actually remove, every project.
    python3 scripts/curation/prune_training_runs.py --apply

    # One project only.
    python3 scripts/curation/prune_training_runs.py --project cars --apply

    # Different retention windows.
    python3 scripts/curation/prune_training_runs.py --keep-last 10 --bakeoff-out-keep-last 5
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from src.config import get_curation_config
from src.config.project_context import bind_project
from src.services.projects.script_binding import (
    abind_script_project,
    add_project_argument,
    script_project_registry,
)
from src.services.training.run_retention import (
    apply_bakeoff_out_prune,
    apply_run_prune,
    plan_bakeoff_out_prune,
    plan_run_prune,
)
from src.services.training.triton_promote import resolve_triton_models_dir


def _run_one_project(args: argparse.Namespace) -> int:
    """Prune training runs + bake-off output for whatever project is
    currently bound. ``train_jobs_dir``/``bakeoff_jobs_dir`` come from
    the bound project's own ``CurationConfig`` view (PROJECT_SCOPED_FIELDS),
    not a raw env var, so each project prunes only its own dir."""
    config = get_curation_config()
    jobs_dir = config.train_jobs_dir
    model_repo = resolve_triton_models_dir()
    bakeoff_jobs_dir = config.bakeoff_jobs_dir
    bakeoff_out_dir = Path(
        os.environ.get('OP_BAKEOFF_OUT_DIR', str(config.state_dir / 'bakeoff_out'))
    )

    run_plan = plan_run_prune(
        jobs_dir=jobs_dir,
        model_repo=model_repo,
        bakeoff_jobs_dir=bakeoff_jobs_dir,
        keep_last=args.keep_last,
    )
    out_plan = plan_bakeoff_out_prune(bakeoff_out_dir, args.bakeoff_out_keep_last)

    print(f'{len(run_plan)} training run(s) planned for removal (keep_last={args.keep_last}):')
    for job_id in run_plan:
        print(f'  {job_id}')
    print(
        f'{len(out_plan)} bake-off output dir(s) planned for removal '
        f'(keep_last={args.bakeoff_out_keep_last}):'
    )
    for path in out_plan:
        print(f'  {path}')

    if not args.apply:
        print('Dry-run only. Pass --apply to remove.')
        return 0

    run_result = apply_run_prune(run_plan, jobs_dir=jobs_dir)
    out_result = apply_bakeoff_out_prune(out_plan)
    print(
        f'removed_runs={run_result["removed_runs"]} removed_run_files={run_result["removed_files"]} '
        f'removed_bakeoff_out={out_result["removed"]}'
    )
    return 0


async def run(args: argparse.Namespace) -> int:
    """Prune every active + archived project (or just ``--project SLUG``
    when given), each under its own binding."""
    if args.project:
        await abind_script_project(args.project)
        return _run_one_project(args)

    registry = script_project_registry()
    await registry.refresh_strict()
    projects = registry.active_projects() + registry.archived_projects()
    rc = 0
    for record in projects:
        print(f'== project {record.slug} ({record.status}) ==')
        with bind_project(record, read_only=record.status == 'archived'):
            rc = _run_one_project(args) or rc
    return rc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('--keep-last', type=int, default=20)
    p.add_argument('--bakeoff-out-keep-last', type=int, default=20)
    g = p.add_mutually_exclusive_group()
    g.add_argument('--dry-run', dest='apply', action='store_false', default=False)
    g.add_argument('--apply', dest='apply', action='store_true')
    add_project_argument(p)
    # Default: every active + archived project, each under its own binding.
    p.set_defaults(project=None)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return asyncio.run(run(args))


if __name__ == '__main__':
    sys.exit(main())
