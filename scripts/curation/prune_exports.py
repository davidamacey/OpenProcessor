#!/usr/bin/env python3
"""Keep-last prune of auto-named export directories (ST-4).

``export_dataset`` (``src/services/curation/export.py``) auto-runs this
best-effort after every successful export, but a deployment that hasn't
exported in a while (or ran with an older API image) can still have a
backlog. This is the operator-facing equivalent: same planning logic
(:mod:`src.services.curation.export_retention`), read-only by default.

Never touches: a custom-named export dir, the ``current`` symlink's
target, or any export dir a training job, a finished run's lineage, or a
queued/running bake-off still references.

Loops over every active AND archived project by default, pruning each
project's own exports under its own binding (projects_plan.md §11 W7).
``--project SLUG`` restricts the run to just that one project.

    # See what would be removed, every project.
    python3 scripts/curation/prune_exports.py

    # Actually remove, every project.
    python3 scripts/curation/prune_exports.py --apply

    # One project only.
    python3 scripts/curation/prune_exports.py --project cars --apply

    # Different retention window.
    python3 scripts/curation/prune_exports.py --keep-last 10
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from src.config import get_curation_config
from src.config.project_context import bind_project
from src.services.curation.export_retention import (
    apply_export_prune,
    collect_export_pins,
    plan_export_prune,
)
from src.services.projects.registry import get_project_registry
from src.services.projects.script_binding import bind_script_project


def _run_one_project(args: argparse.Namespace) -> int:
    """Prune exports for whatever project is currently bound.
    ``train_jobs_dir``/``bakeoff_jobs_dir`` come from the bound
    project's own ``CurationConfig`` view (PROJECT_SCOPED_FIELDS), not
    a raw env var / the global GPU-arbiter config, so each project's
    export pins are computed against its own dirs only."""
    config = get_curation_config()
    export_root = Path(args.export_root) if args.export_root else config.export_root
    jobs_dir = config.train_jobs_dir
    bakeoff_jobs_dir = config.bakeoff_jobs_dir

    pins = collect_export_pins(
        export_root=export_root, jobs_dir=jobs_dir, bakeoff_jobs_dir=bakeoff_jobs_dir
    )
    plan = plan_export_prune(export_root, args.keep_last, pins)

    if not plan:
        print(f'Nothing to prune under {export_root} (keep_last={args.keep_last}).')
        return 0

    print(f'{len(plan)} export dir(s) planned for removal (keep_last={args.keep_last}):')
    for path in plan:
        print(f'  {path}')

    if not args.apply:
        print('Dry-run only. Pass --apply to remove.')
        return 0

    result = apply_export_prune(plan)
    print(f'removed={result["removed"]} removed_bytes={result["removed_bytes"]}')
    return 0


async def run(args: argparse.Namespace) -> int:
    """Prune every active + archived project (or just ``--project SLUG``
    when given), each under its own binding."""
    if args.project:
        bind_script_project(args.project)
        return _run_one_project(args)

    registry = get_project_registry()
    await registry.ensure_fresh()
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
    p.add_argument('--keep-last', type=int, default=get_curation_config().export_keep_last)
    p.add_argument('--export-root', default=None, help='Override CurationConfig.export_root.')
    g = p.add_mutually_exclusive_group()
    g.add_argument('--dry-run', dest='apply', action='store_false', default=False)
    g.add_argument('--apply', dest='apply', action='store_true')
    p.add_argument(
        '--project',
        default=None,
        help='Restrict to one project slug. Default: every active + archived project.',
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    return asyncio.run(run(args))


if __name__ == '__main__':
    sys.exit(main())
