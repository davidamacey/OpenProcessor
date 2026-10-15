"""Training-job control protocol (file-based).

See ``docs/design/curation_design_rationale.md`` for the design
rationale. This module encapsulates the API <-> trainer protocol:

- The API writes ``<job_id>.job.json`` into the shared ``/jobs/`` volume
  to start a run. The trainer container watches the directory.
- The trainer writes ``<job_id>.status.json`` every 5s while running.
- Cancellation is a sentinel ``<job_id>.cancel`` file the API drops next
  to ``job.json``; the trainer checks for it between epochs.
- Progress logs live in ``<job_id>.run.log`` (append-only stdout/stderr).

The router stays thin and delegates everything to the ``write_*`` /
``read_*`` / ``list_*`` helpers below. This keeps the file-format
responsibilities in one place and makes testing trivial -- the entire
protocol is a tmpdir + JSON.

Environment:
    OP_TRAIN_JOBS_DIR
        Override the ``/jobs/`` mount point (test fixtures use a
        ``tmp_path``). Default: ``/jobs``.

Heartbeat semantics:
    The trainer writes ``heartbeat_at`` every status update. If we read
    a status whose ``heartbeat_at`` is more than
    :data:`STALE_HEARTBEAT_SECONDS` old AND ``state in {running,
    starting}``, we surface the run as ``state="lost"`` to the caller.
    The on-disk file is left untouched -- only the in-memory model is
    transformed -- so the trainer can still recover.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections import deque
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from src.config import get_curation_config
from src.core.logging import get_logger
from src.services.training import lineage
from src.services.training.job_files import (
    STALE_HEARTBEAT_SECONDS,
    _atomic_write_json,
    _cancel_path,
    _job_path,
    _log_path,
    _read_json,
    _registry_snapshot_path,
    _resolve_jobs_dir,
    _slug,
    _status_path,
    _validate_job_id,
)
from src.services.training.job_models import TrainJobSpec, TrainJobStatus
from src.services.training.job_wire import _prepare_status_for_wire
from src.utils.timeutil import now_iso


if TYPE_CHECKING:
    from pathlib import Path


logger = get_logger(__name__)


def _pin_registry_snapshot_sync(job_id: str) -> tuple[str | None, str | None]:
    """Snapshot the live class registry to a job-scoped file; return (sha, path).

    ``labels.txt`` is rebuilt from the *live* registry at promote time
    unless pinned, so a class rename between export and promote could
    silently relabel the served model. Pinning a copy + sha256 at submit
    time lets promote use the registry as it was when the run was
    launched. Runs entirely in a thread (sync file I/O) -- see
    :func:`write_job`.

    Best-effort: any failure (missing registry file, unwritable jobs dir)
    returns ``(None, None)`` rather than raising -- a registry hiccup must
    never block ``/curation/train/start``.
    """
    import hashlib

    from src.clients.curation_opensearch.registry import get_class_registry

    try:
        registry = get_class_registry().load()
        content = registry.model_dump_json(indent=2)
    except Exception as exc:
        logger.warning('curation_train_registry_pin_read_failed', job_id=job_id, error=str(exc))
        return None, None
    sha = hashlib.sha256(content.encode('utf-8')).hexdigest()
    snapshot_path = _registry_snapshot_path(job_id)
    try:
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = snapshot_path.with_suffix(snapshot_path.suffix + '.tmp')
        tmp.write_text(content, encoding='utf-8')
        tmp.replace(snapshot_path)
    except OSError as exc:
        logger.warning(
            'curation_train_registry_pin_write_failed',
            job_id=job_id,
            path=str(snapshot_path),
            error=str(exc),
        )
        return sha, None
    return sha, str(snapshot_path)


# =============================================================================
# Public API -- write + read
# =============================================================================


async def write_job(job: TrainJobSpec) -> str:
    """Materialize ``<job_id>.job.json`` and return the assigned ``job_id``.

    If ``job.job_id`` is unset, we generate one of the form
    ``<isoslug>_<family><size>``. Also auto-fills the lineage fields the
    caller normally doesn't set directly: ``dataset_sha`` /
    ``frozen_test_sha`` / ``test_label_sha`` / ``dataset_version_tag`` from
    the export manifest (:func:`~src.services.training.lineage.read_export_identity`),
    ``api_sha`` / ``trainer_image_id`` / ``trainer_image_revision`` from this
    build's own identity (:func:`~src.services.training.lineage.stamp_code_versions`),
    and ``registry_sha``/``registry_snapshot_path`` by pinning the live class
    registry at this exact moment.
    """
    spec = job.model_copy(deep=True)
    if not spec.job_id:
        spec.job_id = f'{_slug()}_{spec.model_family}{spec.model_size}'
    if not spec.submitted_at:
        spec.submitted_at = now_iso()

    cfg = get_curation_config()
    spec.project = cfg.project_slug
    spec.project_export_root = str(cfg.export_root)
    spec.mlflow_experiment = cfg.mlflow_experiment

    # dataset_export_dir is optional on the wire (defaults to the
    # current export -- resolved by _run_preflight, which every /start
    # caller runs first). It can still reach here as None if the caller
    # used force=True to bypass a *blocking* preflight (the block fires
    # precisely when there's no current export and none was given) --
    # fail loudly and specifically here rather than an opaque TypeError
    # a few lines down inside read_export_identity(None).
    if not spec.dataset_export_dir:
        msg = 'dataset_export_dir is required (no current export exists to default to)'
        raise ValueError(msg)

    _validate_job_id(spec.job_id)
    target = _job_path(spec.job_id)

    if target.exists():
        msg = f'job_id already exists: {spec.job_id}'
        raise ValueError(msg)

    # Pin lineage AFTER the duplicate check so a rejected duplicate submit
    # never leaves an orphaned registry-snapshot file behind.
    if spec.dataset_sha is None or spec.frozen_test_sha is None or spec.test_label_sha is None:
        identity = await asyncio.to_thread(lineage.read_export_identity, spec.dataset_export_dir)
        if spec.dataset_sha is None:
            spec.dataset_sha = identity.dataset_sha
        if spec.frozen_test_sha is None:
            spec.frozen_test_sha = identity.frozen_test_sha
        if spec.test_label_sha is None:
            spec.test_label_sha = identity.test_label_sha
        if spec.dataset_version_tag is None:
            spec.dataset_version_tag = identity.version_tag
    if spec.api_sha is None or spec.trainer_image_id is None:
        code_versions = await asyncio.to_thread(lineage.stamp_code_versions)
        if spec.api_sha is None:
            spec.api_sha = code_versions['api_sha']
        if spec.trainer_image_id is None:
            spec.trainer_image_id = code_versions['trainer_image_id']
        if spec.trainer_image_revision is None:
            spec.trainer_image_revision = code_versions['trainer_image_revision']
    if spec.registry_sha is None:
        spec.registry_sha, spec.registry_snapshot_path = await asyncio.to_thread(
            _pin_registry_snapshot_sync, spec.job_id
        )

    payload = spec.model_dump(mode='json', exclude_none=False)
    # The trainer calls back into the API (campaign auto-promote, the
    # auto-quantize bake-off) under this project's prefix; the client never
    # chooses it.
    from src.config.project_context import current_project

    payload['project'] = current_project().record.slug

    await _atomic_write_json(target, payload)
    logger.info(
        'curation_train_job_written',
        job_id=spec.job_id,
        campaign_id=spec.campaign_id,
        family=spec.model_family,
        size=spec.model_size,
        profile=spec.profile,
        path=str(target),
    )
    return spec.job_id


async def write_cancel(job_id: str) -> None:
    """Drop the ``<job_id>.cancel`` sentinel.

    The trainer polls between epochs and raises a ``KeyboardInterrupt``
    when it sees the file. Idempotent -- re-cancelling is a no-op.
    """
    _validate_job_id(job_id)
    target = _cancel_path(job_id)

    def _do() -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        # touch
        target.open('a', encoding='utf-8').close()

    await asyncio.to_thread(_do)
    logger.info('curation_train_cancel_written', job_id=job_id, path=str(target))


def _maybe_lost(status: TrainJobStatus) -> TrainJobStatus:
    """Mark a stale running run as ``lost``."""
    if status.state not in ('running', 'starting'):
        return status
    if not status.heartbeat_at:
        return status
    try:
        hb = datetime.fromisoformat(status.heartbeat_at)
    except ValueError:
        logger.debug('curation_train_heartbeat_unparseable', job_id=status.job_id)
        return status
    if hb.tzinfo is None:
        hb = hb.replace(tzinfo=UTC)
    age = datetime.now(UTC) - hb
    if age > timedelta(seconds=STALE_HEARTBEAT_SECONDS):
        return status.model_copy(update={'state': 'lost'})
    return status


async def read_status(job_id: str) -> TrainJobStatus | None:
    """Return the trainer-written ``status.json`` for ``job_id`` or ``None``.

    A stale heartbeat (over :data:`STALE_HEARTBEAT_SECONDS`) flips
    ``state`` to ``"lost"`` only in the returned object -- the on-disk
    file is left as the trainer wrote it.
    """
    _validate_job_id(job_id)
    raw = await _read_json(_status_path(job_id))
    if raw is None:
        # Fall back to the spec -- the run hasn't started yet but the
        # job.json exists. Show it as ``queued``.
        spec_raw = await _read_json(_job_path(job_id))
        if spec_raw is None:
            return None
        return TrainJobStatus(
            job_id=spec_raw.get('job_id', job_id),
            campaign_id=spec_raw.get('campaign_id'),
            state='queued',
        )
    try:
        status = TrainJobStatus.model_validate(raw)
    except Exception as exc:
        logger.warning(
            'curation_train_status_invalid',
            job_id=job_id,
            error=str(exc),
        )
        return None
    return _prepare_status_for_wire(_maybe_lost(status))


def _list_status_files() -> list[Path]:
    jobs_dir = _resolve_jobs_dir()
    if not jobs_dir.exists():
        return []
    return sorted(
        jobs_dir.glob('*.status.json'),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _list_job_files() -> list[Path]:
    jobs_dir = _resolve_jobs_dir()
    if not jobs_dir.exists():
        return []
    return sorted(
        jobs_dir.glob('*.job.json'),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _load_json_object(path: Path) -> dict[str, Any] | None:
    """``path`` parsed as a JSON object, or ``None`` if unreadable, not JSON or not an object."""
    try:
        with path.open('r', encoding='utf-8') as fh:
            raw = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def _gather_run_rows() -> list[tuple[Path, dict[str, Any], str]]:
    out: list[tuple[Path, dict[str, Any], str]] = []
    jobs_dir = _resolve_jobs_dir()
    if not jobs_dir.exists():
        return out
    # status.json wins where both exist
    seen: set[str] = set()
    for sp in _list_status_files():
        stem = sp.name[: -len('.status.json')]
        raw = _load_json_object(sp)
        if raw is None:
            continue
        seen.add(stem)
        out.append((sp, raw, 'status'))
    for jp in _list_job_files():
        stem = jp.name[: -len('.job.json')]
        if stem in seen:
            continue
        raw = _load_json_object(jp)
        if raw is None:
            continue
        out.append((jp, raw, 'job'))
    return out


def _campaign_id_from_job_file(stem: str) -> str | None:
    """The ``campaign_id`` recorded in ``<stem>.job.json``, if that file has one."""
    spec_path = _resolve_jobs_dir() / f'{stem}.job.json'
    if not spec_path.exists():
        return None
    spec_raw = _load_json_object(spec_path)
    if spec_raw is not None and spec_raw.get('campaign_id'):
        typed: str | None = spec_raw['campaign_id']
        return typed
    return None


def _status_row_to_wire(path: Path, raw: dict[str, Any]) -> TrainJobStatus | None:
    try:
        status = TrainJobStatus.model_validate(raw)
    except Exception as exc:
        logger.warning('curation_train_status_invalid', error=str(exc))
        return None
    # If the trainer didn't carry through ``campaign_id`` in the
    # status, re-hydrate it from the matching job.json so the
    # campaign-cancel path can find sibling runs.
    if status.campaign_id is None:
        campaign_id = _campaign_id_from_job_file(path.name[: -len('.status.json')])
        if campaign_id is not None:
            status = status.model_copy(update={'campaign_id': campaign_id})
    return _prepare_status_for_wire(_maybe_lost(status))


async def list_runs(limit: int = 50, offset: int = 0) -> list[TrainJobStatus]:
    """Return up to ``limit`` past + active runs, newest first.

    Source of truth for the listing is the ``status.json`` file (so a
    queued-but-never-started run shows up too). For each ``status.json``
    we apply the same heartbeat-staleness rule as :func:`read_status`.

    Jobs that have a ``job.json`` but no ``status.json`` (i.e. queued in
    the trainer but never picked up) are listed with ``state='queued'``
    so the UI can display them.
    """
    if limit <= 0:
        return []

    rows = await asyncio.to_thread(_gather_run_rows)
    rows.sort(key=lambda t: t[0].stat().st_mtime, reverse=True)
    rows = rows[offset : offset + limit]

    results: list[TrainJobStatus] = []
    for path, raw, kind in rows:
        if kind == 'status':
            status = _status_row_to_wire(path, raw)
            if status is not None:
                results.append(status)
        else:
            results.append(
                TrainJobStatus(
                    job_id=raw.get('job_id', path.stem),
                    campaign_id=raw.get('campaign_id'),
                    state='queued',
                )
            )
    return results


async def get_active_job() -> TrainJobStatus | None:
    """Return the single non-terminal run if exactly one exists.

    Used by the preflight ``active_run`` check and the legacy
    ``GET /curation/train/status`` endpoint. Raises ``RuntimeError`` if
    the filesystem advertises more than one active run -- that's a bug in
    the trainer (we only ever write one job at a time) and the caller
    should surface it as a 500.
    """
    runs = await list_runs(limit=50)
    active = [r for r in runs if r.state in ('queued', 'starting', 'running', 'exporting')]
    if not active:
        # Most-recent finished/failed run for the legacy /status endpoint.
        return runs[0] if runs else None
    if len(active) > 1:
        ids = ', '.join(r.job_id for r in active)
        msg = f'multiple active runs found: {ids}'
        raise RuntimeError(msg)
    return active[0]


async def tail_run_log(job_id: str, lines: int = 200) -> list[str]:
    """Return the last ``lines`` lines of ``<job_id>.run.log``.

    Reads the file in a chunk-tail loop so a multi-MB log doesn't get
    fully buffered into memory.
    """
    _validate_job_id(job_id)
    if lines <= 0:
        return []

    path = _log_path(job_id)

    def _do() -> list[str]:
        if not path.exists():
            return []
        try:
            with path.open('rb') as fh:
                fh.seek(0, os.SEEK_END)
                file_size = fh.tell()
                block_size = 8192
                blocks: deque[bytes] = deque()
                read = 0
                count = 0
                while read < file_size and count <= lines:
                    chunk_size = min(block_size, file_size - read)
                    fh.seek(file_size - read - chunk_size)
                    block = fh.read(chunk_size)
                    blocks.appendleft(block)
                    read += chunk_size
                    count = b''.join(blocks).count(b'\n')
                buf = b''.join(blocks)
        except OSError as exc:
            logger.warning('curation_train_log_read_failed', job_id=job_id, error=str(exc))
            return []

        text = buf.decode('utf-8', errors='replace').splitlines()
        return text[-lines:]

    return await asyncio.to_thread(_do)
