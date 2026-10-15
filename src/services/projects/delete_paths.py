"""Path validation and removal for the filesystem directories a project owns.

Split out of ``delete.py``; every path is checked to sit under a
project-scoped directory before anything is removed.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

from src.config.env import env_path
from src.core.logging import get_logger
from src.services.rejections import rejection


if TYPE_CHECKING:
    from pathlib import Path

    from src.config.projects import ProjectRecord


logger = get_logger(__name__)


def _require_project_scoped_path(path: Path, shared_root: Path, slug: str) -> None:
    """m1 / P3F pass-3 m-a: require ``path`` to resolve to
    ``<shared_root>/<slug>`` itself, or a proper descendant of it --
    never ``shared_root`` itself (a corrupted/hand-edited registry
    ``resources`` record pointing straight at the multi-project root
    would otherwise let a delete wipe every sibling project's dir), and
    never a sibling project's own ``<shared_root>/<other_slug>`` tree.

    The prior guard (``_rm_dir_guarded``, ``_path_within``) accepted
    ``path == shared_root`` for 6 of the project's 8 dirs -- only
    ``train_jobs_dir``/``autolabel_dir`` got a real per-slug check. This
    is the one guard every dir now goes through, each with its own
    independently-computed ``shared_root`` (never derived from the path
    being checked), and it always raises the plan's ``path_escape``
    code, never ``internal_isolation_error``."""
    project_root = (shared_root / slug).resolve()
    try:
        path.resolve().relative_to(project_root)
    except ValueError:
        logger.error('project_delete_path_escape', path=str(path), expected_root=str(project_root))
        raise rejection(500, 'path_escape', f'refusing to delete outside {project_root}') from None


def _project_scoped_dirs(record: ProjectRecord) -> list[tuple[Path, Path]]:
    """Every one of the project's own dirs, paired with the shared
    root each must resolve strictly inside (§4 step 6: "a path outside
    refuses"). A new project's dirs are siblings under
    ``OP_PROJECTS_DATA_ROOT`` (exports, class registry, bakeoff eval) or
    under the deployment ``state_dir``/jobs roots (uploads, job dirs,
    autolabel state) -- never under ``default``'s own dirs, which
    :func:`resources_for_new` never nests anything into."""
    from src.config.curation_accessors import base_curation_config
    from src.config.projects import projects_data_root, trainer_jobs_root

    base = base_curation_config()
    data_root = projects_data_root()
    state_projects_root = base.state_dir / 'projects'
    autolabel_projects_root = env_path('OP_AUTO_LABEL_STATE_DIR', '/jobs/auto_label') / 'projects'
    from src.services.curation.bulk_ingest.limits import bulk_ingest_base_dir

    bulk_ingest_projects_root = bulk_ingest_base_dir() / 'projects'
    reprocess_projects_root = env_path('OP_REPROCESS_JOBS_DIR', '/jobs/reprocess') / 'projects'
    return [
        (record.resources.export_root, data_root),
        (record.resources.class_registry_path.parent, data_root),
        (record.resources.bakeoff_eval_root, data_root),
        (record.resources.project_state_dir, state_projects_root),
        (record.resources.upload_root, state_projects_root),
        (record.resources.bakeoff_jobs_dir, state_projects_root),
        (record.resources.train_jobs_dir, trainer_jobs_root() / 'projects'),
        (record.resources.autolabel_dir, autolabel_projects_root),
        # Bulk-ingest runs, ingest sources (definitions and queues) and reprocess jobs sit
        # under the shared jobs volume, keyed by slug, not in the project's own resources.
        (bulk_ingest_projects_root / record.slug, bulk_ingest_projects_root),
        (reprocess_projects_root / record.slug, reprocess_projects_root),
    ]


def _validate_delete_paths(record: ProjectRecord) -> None:
    """P3F pass-3 m-a: validate every one of the project's 8 dirs BEFORE
    any irreversible step runs. The prior ordering ran this check (for
    the 2 dirs it covered) only from inside ``_delete_dirs``, itself
    called AFTER ``_delete_indexes`` had already irreversibly deleted
    the project's OpenSearch indexes -- a ``path_escape`` raised there
    left the record wedged ``deleting`` forever (every retry hits the
    same escape again) with the indexes already gone. Called as a
    preflight in :func:`delete_project_finish`, before the drain wait,
    model unload or index delete."""
    for path, shared_root in _project_scoped_dirs(record):
        _require_project_scoped_path(path, shared_root, record.slug)


async def _delete_dirs(record: ProjectRecord) -> None:
    """Remove every per-project dir. Paths are already validated by
    :func:`_validate_delete_paths` earlier in
    :func:`delete_project_finish`; re-validate here too (cheap, and this
    function has its own unit-test callers) rather than trusting that
    nothing mutated ``record.resources`` in between."""
    _validate_delete_paths(record)
    for path, _shared_root in _project_scoped_dirs(record):
        shutil.rmtree(path, ignore_errors=True)
