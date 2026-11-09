"""ST-4: keep-last retention for auto-named export directories.

``export_dataset`` (``src/services/curation/export.py``) had no retention at
all -- every export accumulated forever under ``CurationConfig.export_root``.
This module plans (and, best-effort, applies) a keep-last prune that never
touches:

* a **custom-named** export dir (e.g. an operator-supplied ``export_dir=``,
  or an example profile's fixed name) -- only auto-named,
  ``YYYYMMDDTHHMMSSZ``-stamped dirs are ever candidates;
* the resolved target of the ``current`` symlink;
* any export dir a training run (queued, running or finished) or a queued/
  running bake-off job still references (a "pin").

Planning and applying are separate on purpose: ``plan_export_prune`` is
pure (given the filesystem state) and cheap to unit-test; only
``apply_export_prune`` touches disk.
"""

from __future__ import annotations

import contextlib
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from src.config.project_context import current_project_id
from src.core.logging import get_logger
from src.db.repos import training_runs as runs_repo
from src.db.session import sync_project_tx, sync_session_factory
from src.services.training.bakeoff_pins import active_bakeoff_specs


if TYPE_CHECKING:
    from src.config import CurationConfig


logger = get_logger(__name__)

# Auto-named export dirs only (export.py:494's datetime.now(UTC).strftime
# format). A custom-named dir (an operator's --export-dir, an example
# profile's fixed name) never matches this and is never a prune candidate.
EXPORT_DIR_RE = re.compile(r'^\d{8}T\d{6}Z$')


def _resolve_or_none(path_str: str) -> Path | None:
    try:
        return Path(path_str).resolve()
    except OSError:
        return None


def _run_export_dirs() -> set[str]:
    with (
        sync_session_factory()() as session,
        sync_project_tx(session, current_project_id()),
    ):
        return runs_repo.export_dirs(session)


def collect_export_pins(*, export_root: Path) -> set[Path]:
    """Every export directory that must never be pruned.

    Sources (a database error propagates, so a prune never proceeds without its pins):

    * the resolved target of ``<export_root>/current``;
    * every export directory a training run of the bound project was queued with or whose
      manifest records as its lineage, read from ``training_runs``;
    * every dataset ``path`` (and the quantize calibration dataset) in the params of an
      active ``bakeoff`` job, read from PostgreSQL.
    """
    pins: set[Path] = set()

    current = export_root / 'current'
    if current.is_symlink() or current.exists():
        with contextlib.suppress(OSError):
            pins.add(current.resolve())

    for export_dir in _run_export_dirs():
        resolved = _resolve_or_none(export_dir)
        if resolved is not None:
            pins.add(resolved)

    for spec in active_bakeoff_specs():
        paths = [ds.get('path') for ds in spec.get('datasets') or [] if isinstance(ds, dict)]
        quantize = spec.get('quantize')
        if isinstance(quantize, dict):
            paths.append(quantize.get('calib_dataset'))
        for path in paths:
            resolved = _resolve_or_none(path) if path else None
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

    Pins come from the bound project's training runs and the active bake-offs in PostgreSQL. A
    no-op when ``export_keep_last <= 0``.
    """
    keep_last = config.export_keep_last
    if keep_last <= 0:
        return
    pins = collect_export_pins(export_root=config.export_root)
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
