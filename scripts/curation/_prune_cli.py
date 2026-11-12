"""The command line the keep-last prune scripts share: every active and archived project
under its own binding (or one ``--project``), dry-run unless ``--apply``."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from src.config.project_context import bind_project
from src.services.projects.script_binding import (
    abind_script_project,
    add_project_argument,
    script_project_registry,
)


if TYPE_CHECKING:
    from collections.abc import Callable


def new_prune_parser(description: str | None) -> argparse.ArgumentParser:
    """A parser with the dry-run/apply group and ``--project`` (default: every project)."""
    p = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument('--dry-run', dest='apply', action='store_false', default=False)
    g.add_argument('--apply', dest='apply', action='store_true')
    add_project_argument(p)
    p.set_defaults(project=None)
    return p


async def run_over_projects(
    args: argparse.Namespace, run_one_project: Callable[[argparse.Namespace], int]
) -> int:
    """``run_one_project(args)`` for ``--project SLUG``, else for every active + archived
    project, each under its own binding (archived ones read-only)."""
    if args.project:
        await abind_script_project(args.project)
        return run_one_project(args)

    registry = script_project_registry()
    await registry.refresh_strict()
    projects = registry.active_projects() + registry.archived_projects()
    rc = 0
    for record in projects:
        print(f'== project {record.slug} ({record.status}) ==')
        with bind_project(record, read_only=record.status == 'archived'):
            rc = run_one_project(args) or rc
    return rc
