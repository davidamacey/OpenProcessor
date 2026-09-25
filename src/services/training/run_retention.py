"""ST-4: keep-last retention for finished training runs.

Training job state (``<job_id>.job.json`` / ``.status.json`` /
``.manifest.json`` / ``.run.log`` / ``.cancel`` / ``.registry_snapshot.json``)
under ``OP_TRAIN_JOBS_DIR`` accumulates forever with nothing to clean it up.
This module plans (and, best-effort, applies) a keep-last prune that never
touches:

* a **non-terminal** run (``queued``/``starting``/``running``/``exporting``);
* a run promoted to Triton (``job_id`` appears in any
  ``<model_repo>/*/promote.json`` -- written by
  ``src.services.training.triton_promote``);
* a run referenced by a queued/running bake-off job (``job_id`` appears as
  a ``models[].run_id`` in any ``<bakeoff_jobs_dir>/*.job.json``).

Job ids are ISO-timestamp-prefixed (see ``src.services.training.jobs``'s
``JOB_ID_RE``), so lexicographic sort on the id string is chronological --
no need to parse ``started_at``/mtimes for ordering.
"""

from __future__ import annotations

import contextlib
import json
import shutil
from typing import TYPE_CHECKING, Any

from src.core.logging import get_logger


if TYPE_CHECKING:
    from pathlib import Path


logger = get_logger(__name__)

# Mirrors src.services.training.jobs.TRAIN_STATES' terminal subset. Kept as
# a literal tuple here (rather than importing TRAIN_STATES) so this module
# stays a light, dependency-free filesystem scanner -- importing
# jobs.py would pull in its Pydantic status model and OpenSearch-adjacent
# imports for something that only needs four literal strings.
TERMINAL_STATES = frozenset({'finished', 'failed', 'cancelled', 'lost'})

_JOB_FILE_SUFFIXES = (
    '.job.json',
    '.status.json',
    '.manifest.json',
    '.run.log',
    '.cancel',
    '.registry_snapshot.json',
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open('r', encoding='utf-8') as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug('run_retention_read_failed', path=str(path), error=str(exc))
        return {}
    return data if isinstance(data, dict) else {}


def list_runs(jobs_dir: Path) -> dict[str, dict[str, Any]]:
    """Every job_id -> its status.json content (or ``{'state': None}`` for a
    job.json with no status yet, i.e. still queued)."""
    if not jobs_dir.is_dir():
        return {}
    runs: dict[str, dict[str, Any]] = {}
    for status_file in jobs_dir.glob('*.status.json'):
        job_id = status_file.name[: -len('.status.json')]
        runs[job_id] = _read_json(status_file)
    for job_file in jobs_dir.glob('*.job.json'):
        job_id = job_file.name[: -len('.job.json')]
        runs.setdefault(job_id, {'state': 'queued'})
    return runs


def collect_promoted_job_ids(model_repo: Path) -> set[str]:
    """Every ``job_id`` recorded in a served model's ``promote.json``."""
    if not model_repo.is_dir():
        return set()
    ids: set[str] = set()
    for promote_file in model_repo.glob('*/promote.json'):
        job_id = _read_json(promote_file).get('job_id')
        if job_id:
            ids.add(str(job_id))
    return ids


def collect_active_bakeoff_run_ids(bakeoff_jobs_dir: Path) -> set[str]:
    """Every training ``run_id`` a queued/running bake-off job still scores."""
    if not bakeoff_jobs_dir.is_dir():
        return set()
    ids: set[str] = set()
    for job_file in bakeoff_jobs_dir.glob('*.job.json'):
        for model in _read_json(job_file).get('models') or []:
            run_id = model.get('run_id') if isinstance(model, dict) else None
            if run_id:
                ids.add(str(run_id))
    return ids


def plan_run_prune(
    *,
    jobs_dir: Path,
    model_repo: Path,
    bakeoff_jobs_dir: Path,
    keep_last: int,
) -> list[str]:
    """job_ids to remove: terminal runs older than the newest ``keep_last``
    (across every state), skipping anything promoted or bake-off-active.

    ``keep_last <= 0`` keeps everything (returns an empty plan).
    """
    if keep_last <= 0:
        return []

    runs = list_runs(jobs_dir)
    if not runs:
        return []

    ordered = sorted(runs, reverse=True)  # newest job_id first
    beyond_keep_last = ordered[keep_last:]

    promoted = collect_promoted_job_ids(model_repo)
    bakeoff_active = collect_active_bakeoff_run_ids(bakeoff_jobs_dir)

    plan: list[str] = []
    for job_id in beyond_keep_last:
        state = runs[job_id].get('state')
        if state not in TERMINAL_STATES:
            continue
        if job_id in promoted or job_id in bakeoff_active:
            continue
        plan.append(job_id)
    return plan


def apply_run_prune(job_ids: list[str], *, jobs_dir: Path) -> dict[str, int]:
    """Delete every job-state file for each planned ``job_id``.

    Best-effort per file -- a missing/unremovable file is skipped, never
    raised. Does not touch MLflow runs or the trainer's own artifact
    directory (``OP_TRAIN_RUNS_ROOT``); those are a separate concern
    (MLflow retention is owner-decision, W7).
    """
    removed_files = 0
    removed_runs = 0
    for job_id in job_ids:
        any_removed = False
        for suffix in _JOB_FILE_SUFFIXES:
            path = jobs_dir / f'{job_id}{suffix}'
            if not path.exists():
                continue
            with contextlib.suppress(OSError):
                path.unlink()
                removed_files += 1
                any_removed = True
        if any_removed:
            removed_runs += 1
            logger.info('run_pruned', job_id=job_id)
    return {'removed_runs': removed_runs, 'removed_files': removed_files}


def plan_bakeoff_out_prune(out_dir: Path, keep_last: int) -> list[Path]:
    """Bake-off output dirs beyond ``keep_last``, newest-name-first.

    No pin logic: bake-off output is scored comparison output, not a
    training artifact anything else references by path.
    """
    if keep_last <= 0 or not out_dir.is_dir():
        return []
    candidates = sorted(
        (d for d in out_dir.iterdir() if d.is_dir()), key=lambda p: p.name, reverse=True
    )
    return candidates[keep_last:]


def apply_bakeoff_out_prune(paths: list[Path]) -> dict[str, int]:
    removed = 0
    for path in paths:
        try:
            shutil.rmtree(path)
        except OSError as exc:
            logger.warning('bakeoff_out_prune_failed', path=str(path), error=str(exc))
            continue
        removed += 1
        logger.info('bakeoff_out_pruned', path=str(path))
    return {'removed': removed}


__all__ = [
    'TERMINAL_STATES',
    'apply_bakeoff_out_prune',
    'apply_run_prune',
    'collect_active_bakeoff_run_ids',
    'collect_promoted_job_ids',
    'list_runs',
    'plan_bakeoff_out_prune',
    'plan_run_prune',
]
