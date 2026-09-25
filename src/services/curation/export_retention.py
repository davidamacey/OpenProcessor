"""ST-4: keep-last retention for auto-named export directories.

``export_dataset`` (``src/services/curation/export.py``) had no retention at
all -- every export accumulated forever under ``CurationConfig.export_root``.
This module plans (and, best-effort, applies) a keep-last prune that never
touches:

* a **custom-named** export dir (e.g. an operator-supplied ``export_dir=``,
  or an example profile's fixed name) -- only auto-named,
  ``YYYYMMDDTHHMMSSZ``-stamped dirs are ever candidates;
* the resolved target of the ``current`` symlink;
* any export dir a training job, a finished run's lineage, or a queued/
  running bake-off job still references (a "pin").

Planning and applying are separate on purpose: ``plan_export_prune`` is
pure (given the filesystem state) and cheap to unit-test; only
``apply_export_prune`` touches disk.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.core.logging import get_logger


if TYPE_CHECKING:
    from src.config import CurationConfig


logger = get_logger(__name__)

# Auto-named export dirs only (export.py:494's datetime.now(UTC).strftime
# format). A custom-named dir (an operator's --export-dir, an example
# profile's fixed name) never matches this and is never a prune candidate.
EXPORT_DIR_RE = re.compile(r'^\d{8}T\d{6}Z$')


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open('r', encoding='utf-8') as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug('export_retention_read_failed', path=str(path), error=str(exc))
        return {}
    return data if isinstance(data, dict) else {}


def _resolve_or_none(path_str: str) -> Path | None:
    try:
        return Path(path_str).resolve()
    except OSError:
        return None


def collect_export_pins(
    *,
    export_root: Path,
    jobs_dir: Path,
    bakeoff_jobs_dir: Path,
) -> set[Path]:
    """Every export directory that must never be pruned.

    Sources, each best-effort (a missing/unreadable dir or file is simply
    skipped, never raised):

    * the resolved target of ``<export_root>/current``;
    * every ``dataset_export_dir`` in ``<jobs_dir>/*.job.json`` (a training
      job's export, queued or already run);
    * every ``<jobs_dir>/*.manifest.json``'s ``lineage.export_dir`` (a
      finished run's export, which may differ from its job spec's if the
      export was re-pointed after submission);
    * every dataset ``path`` in ``<bakeoff_jobs_dir>/*.job.json`` (a
      queued/running bake-off's eval datasets).
    """
    pins: set[Path] = set()

    current = export_root / 'current'
    if current.is_symlink() or current.exists():
        with contextlib.suppress(OSError):
            pins.add(current.resolve())

    if jobs_dir.is_dir():
        for job_file in jobs_dir.glob('*.job.json'):
            export_dir = _read_json(job_file).get('dataset_export_dir')
            if export_dir:
                resolved = _resolve_or_none(export_dir)
                if resolved is not None:
                    pins.add(resolved)

        for manifest_file in jobs_dir.glob('*.manifest.json'):
            lineage = _read_json(manifest_file).get('lineage') or {}
            export_dir = lineage.get('export_dir') if isinstance(lineage, dict) else None
            if export_dir:
                resolved = _resolve_or_none(export_dir)
                if resolved is not None:
                    pins.add(resolved)

    if bakeoff_jobs_dir.is_dir():
        for bjob_file in bakeoff_jobs_dir.glob('*.job.json'):
            datasets = _read_json(bjob_file).get('datasets') or []
            for ds in datasets:
                path = ds.get('path') if isinstance(ds, dict) else None
                if path:
                    resolved = _resolve_or_none(path)
                    if resolved is not None:
                        pins.add(resolved)

    return pins


def plan_export_prune(
    export_root: Path,
    keep_last: int,
    pins: set[Path],
) -> list[Path]:
    """Auto-named export dirs to remove: sorted newest-name-first, skip the
    first ``keep_last`` and anything pinned. ``keep_last <= 0`` keeps all
    (returns an empty plan)."""
    if keep_last <= 0 or not export_root.is_dir():
        return []

    candidates = sorted(
        (
            d
            for d in export_root.iterdir()
            if d.is_dir() and not d.is_symlink() and EXPORT_DIR_RE.match(d.name)
        ),
        key=lambda p: p.name,
        reverse=True,
    )
    resolved_pins = set()
    for p in pins:
        with contextlib.suppress(OSError):
            resolved_pins.add(p.resolve())

    prunable = candidates[keep_last:]
    return [d for d in prunable if d.resolve() not in resolved_pins]


def apply_export_prune(paths: list[Path]) -> dict[str, int]:
    """``shutil.rmtree`` each planned dir; log name + size. Never raises --
    a per-dir failure is logged and the loop continues."""
    removed = 0
    removed_bytes = 0
    for path in paths:
        try:
            size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        except OSError:
            size = 0
        try:
            shutil.rmtree(path)
        except OSError as exc:
            logger.warning('export_prune_failed', path=str(path), error=str(exc))
            continue
        removed += 1
        removed_bytes += size
        logger.info('export_pruned', path=str(path), bytes=size)
    return {'removed': removed, 'removed_bytes': removed_bytes}


def prune_exports_after_write(config: CurationConfig) -> None:
    """Synchronous keep-last prune, called after a successful export
    (``GenericYoloExportService.export_dataset``, via ``asyncio.to_thread``).

    Resolves the job/bake-off job dirs itself so the caller only needs
    ``CurationConfig``. A no-op when ``export_keep_last <= 0``.
    """
    from src.config import get_gpu_arbiter_config

    keep_last = config.export_keep_last
    if keep_last <= 0:
        return
    jobs_dir = Path(os.environ.get('OP_TRAIN_JOBS_DIR', '/jobs'))
    bakeoff_jobs_dir = Path(get_gpu_arbiter_config().bakeoff_jobs_dir)
    pins = collect_export_pins(
        export_root=config.export_root, jobs_dir=jobs_dir, bakeoff_jobs_dir=bakeoff_jobs_dir
    )
    plan = plan_export_prune(config.export_root, keep_last, pins)
    if not plan:
        return
    result = apply_export_prune(plan)
    logger.info(
        'export_retention_pruned',
        removed=result['removed'],
        removed_bytes=result['removed_bytes'],
        keep_last=keep_last,
    )


__all__ = [
    'EXPORT_DIR_RE',
    'apply_export_prune',
    'collect_export_pins',
    'plan_export_prune',
    'prune_exports_after_write',
]
