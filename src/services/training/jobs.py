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
from pathlib import Path
from typing import Any

from src.config import get_curation_config
from src.core.logging import get_logger
from src.services.training import lineage
from src.services.training.job_files import (
    STALE_HEARTBEAT_SECONDS,
    _atomic_write_json,
    _cancel_path,
    _job_path,
    _log_path,
    _manifest_path,
    _now_iso,
    _read_json,
    _registry_snapshot_path,
    _resolve_jobs_dir,
    _slug,
    _status_path,
    _validate_job_id,
)
from src.services.training.job_models import TrainCampaignSpec, TrainJobSpec, TrainJobStatus
from src.services.training.job_wire import (
    RUN_ARTIFACT_WHITELIST,
    _prepare_status_for_wire,
    _public_mlflow_url,
    _rewrite_eval_for_wire,
)


logger = get_logger(__name__)


async def read_manifest(job_id: str) -> dict[str, Any] | None:
    """Return the parsed run manifest or None if absent.

    The trainer writes the manifest at job-end. The promote endpoint reads
    + updates the ``promoted_to`` field so callers can trace which Triton
    model name a checkpoint shipped under. The served ``results.eval`` /
    ``results.mlflow_run_url`` go through the same wire rewrite as
    ``TrainJobStatus`` (see ``_rewrite_eval_for_wire``/``_public_mlflow_url``)
    -- the manifest is a raw dict, not that model, so it isn't covered by
    ``_prepare_status_for_wire`` automatically.
    """
    _validate_job_id(job_id)
    path = _manifest_path(job_id)

    def _do() -> dict[str, Any] | None:
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    manifest = await asyncio.to_thread(_do)
    if manifest is None:
        return None
    results = manifest.get('results')
    if isinstance(results, dict):
        results = dict(results)
        results['eval'] = _rewrite_eval_for_wire(results.get('eval'), job_id)
        results['mlflow_run_url'] = _public_mlflow_url(
            results.get('mlflow_run_id'), results.get('mlflow_experiment_id')
        )
        manifest = {**manifest, 'results': results}
    return manifest


async def resolve_run_dir(job_id: str) -> Path | None:
    """Resolve the on-disk training-run directory for ``job_id``, best-effort.

    Used only by :func:`read_artifact` to locate metrics/plot files a
    finished (or partially finished) run wrote -- never exposed on the wire
    itself. Derived from the *raw* (unrewritten) ``status.json``:
    ``checkpoint_path`` (``<run_dir>/weights/best.pt``) first, falling back
    to ``eval.confusion_matrix_path`` for a run that failed before export.
    ``None`` if neither is on disk yet.
    """
    _validate_job_id(job_id)
    raw = await _read_json(_status_path(job_id))
    if raw is None:
        return None
    ckpt = raw.get('checkpoint_path')
    if isinstance(ckpt, str) and ckpt:
        return Path(ckpt).resolve().parent.parent
    eval_block = raw.get('eval')
    if isinstance(eval_block, dict):
        cm_path = eval_block.get('confusion_matrix_path')
        if isinstance(cm_path, str) and cm_path:
            return Path(cm_path).resolve().parent
    return None


async def read_artifact(job_id: str, name: str) -> Path | None:
    """Return the absolute path of a whitelisted run artifact, or ``None``.

    ``None`` covers every "don't serve this" case alike (name not in
    :data:`RUN_ARTIFACT_WHITELIST`, unresolvable run dir, missing file, or a
    resolved path that escaped the run dir) -- the router turns any of them
    into a plain 404 so a probe can't learn more than "nothing there."
    """
    _validate_job_id(job_id)
    if name not in RUN_ARTIFACT_WHITELIST:
        return None
    run_dir = await resolve_run_dir(job_id)
    if run_dir is None:
        return None

    def _do() -> Path | None:
        candidate = (run_dir / name).resolve()
        if candidate.parent != run_dir or not candidate.is_file():
            return None
        return candidate

    return await asyncio.to_thread(_do)


async def read_job_spec(job_id: str) -> dict[str, Any] | None:
    """Return the raw ``<job_id>.job.json`` dict, or None if absent.

    Unlike :func:`read_manifest` (trainer-written, best-effort, only exists
    after a run finishes), ``job.json`` is written synchronously by
    :func:`write_job` and always available once a job has been submitted.
    Used by promote to find the registry snapshot pinned at submit time,
    since older/failed runs may lack a manifest entirely.
    """
    _validate_job_id(job_id)
    return await _read_json(_job_path(job_id))


async def stamp_manifest_promotion(
    job_id: str,
    *,
    triton_name: str,
    promoted_at: str,
    force_used: bool = False,
    gate_report: dict[str, Any] | None = None,
) -> bool:
    """Set ``manifest.promoted_to`` after a successful Triton handoff.

    ``force_used`` + ``gate_report`` record whether the promote gate was
    bypassed and, if so, what it found, so a forced promote is traceable
    from the manifest alone.

    Returns False if the manifest doesn't exist yet (older runs without
    one) -- a legitimate, non-error case. Raises on a real failure (corrupt
    JSON, disk/permission error) instead of swallowing it -- the caller
    (the promote router) distinguishes "nothing to stamp" (False, no
    exception) from "stamping broke" (exception) and surfaces the latter
    loudly instead of a silent warn-and-forget.
    """
    _validate_job_id(job_id)
    path = _manifest_path(job_id)

    def _do() -> bool:
        if not path.is_file():
            return False
        payload = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(payload, dict):
            msg = f'manifest for job {job_id!r} is not a JSON object'
            raise ValueError(msg)
        payload['promoted_to'] = {
            'triton_model_name': triton_name,
            'promoted_at': promoted_at,
            'force_used': force_used,
            'gate_report': gate_report,
        }
        tmp = path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
        tmp.replace(path)
        return True

    return await asyncio.to_thread(_do)


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
        spec.submitted_at = _now_iso()

    cfg = get_curation_config()
    spec.project = cfg.project_slug
    spec.project_export_root = str(cfg.export_root)
    spec.mlflow_experiment = cfg.mlflow_experiment

    # F-73: dataset_export_dir is optional on the wire (defaults to the
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

    def _gather() -> list[tuple[Path, dict[str, Any], str]]:
        out: list[tuple[Path, dict[str, Any], str]] = []
        jobs_dir = _resolve_jobs_dir()
        if not jobs_dir.exists():
            return out
        # status.json wins where both exist
        seen: set[str] = set()
        for sp in _list_status_files():
            stem = sp.name[: -len('.status.json')]
            try:
                with sp.open('r', encoding='utf-8') as fh:
                    raw = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(raw, dict):
                continue
            seen.add(stem)
            out.append((sp, raw, 'status'))
        for jp in _list_job_files():
            stem = jp.name[: -len('.job.json')]
            if stem in seen:
                continue
            try:
                with jp.open('r', encoding='utf-8') as fh:
                    raw = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(raw, dict):
                continue
            out.append((jp, raw, 'job'))
        return out

    rows = await asyncio.to_thread(_gather)
    rows.sort(key=lambda t: t[0].stat().st_mtime, reverse=True)
    rows = rows[offset : offset + limit]

    results: list[TrainJobStatus] = []
    for path, raw, kind in rows:
        if kind == 'status':
            try:
                status = TrainJobStatus.model_validate(raw)
            except Exception as exc:
                logger.warning('curation_train_status_invalid', error=str(exc))
                continue
            # If the trainer didn't carry through ``campaign_id`` in the
            # status, re-hydrate it from the matching job.json so the
            # campaign-cancel path can find sibling runs.
            if status.campaign_id is None:
                stem = path.name[: -len('.status.json')]
                spec_path = _resolve_jobs_dir() / f'{stem}.job.json'
                if spec_path.exists():
                    try:
                        with spec_path.open('r', encoding='utf-8') as fh:
                            spec_raw = json.load(fh)
                        if isinstance(spec_raw, dict) and spec_raw.get('campaign_id'):
                            status = status.model_copy(
                                update={'campaign_id': spec_raw['campaign_id']}
                            )
                    except (OSError, json.JSONDecodeError):
                        pass
            results.append(_prepare_status_for_wire(_maybe_lost(status)))
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


# =============================================================================
# Campaigns
# =============================================================================


async def write_campaign(campaign: TrainCampaignSpec) -> tuple[str, list[str]]:
    """Materialize one ``job.json`` per ``runs[]`` entry in order.

    Returns ``(campaign_id, [job_id, ...])``. The trainer processes the
    files in the order they're written; we write smallest-to-largest so
    the user gets fast feedback first.
    """
    spec = campaign.model_copy(deep=True)
    if not spec.campaign_id:
        spec.campaign_id = f'{_slug()}_campaign'
    _validate_job_id(spec.campaign_id)

    valid_profiles = {'probe', 'nano', 'small', 'medium', 'large', 'xlarge', 'custom'}
    job_ids: list[str] = []
    last_idx = len(spec.runs) - 1
    for idx, run in enumerate(spec.runs):
        # Narrow ``profile`` to the literal set TrainJobSpec accepts;
        # fall through to ``custom`` for unknown names so the trainer can
        # still receive a hyperparameter override-only spec.
        profile_value: Any = run.profile if run.profile in valid_profiles else 'custom'
        job_spec = TrainJobSpec(
            job_id=f'{spec.campaign_id}_run{idx:02d}_{run.profile}',
            campaign_id=spec.campaign_id,
            submitted_by=spec.submitted_by,
            model_family='yolo26',
            model_size=run.model_size or 'm',
            profile=profile_value,
            dataset_export_dir=spec.dataset_export_dir,
            include_classes=spec.include_classes,
            single_cls=spec.single_cls,
            cuda_visible_devices=spec.cuda_visible_devices,
            hyperparameters=run.hyperparameters,
            augmentation=spec.augmentation,
            # Carry campaign-wide policy onto every per-job spec so the
            # trainer can act on them between runs (auto-skip +
            # auto-promote).
            stop_when=spec.stop_when,
            auto_promote_best=spec.auto_promote_best,
            is_last_in_campaign=(idx == last_idx),
        )
        jid = await write_job(job_spec)
        job_ids.append(jid)

    logger.info(
        'curation_train_campaign_written',
        campaign_id=spec.campaign_id,
        n_runs=len(job_ids),
    )
    return spec.campaign_id, job_ids


async def cancel_campaign(campaign_id: str) -> int:
    """Cancel every job belonging to ``campaign_id``.

    We walk every ``job.json`` in the directory, match on the embedded
    ``campaign_id``, and drop a cancel sentinel for non-terminal runs.
    Returns the count of sentinels written.
    """
    _validate_job_id(campaign_id)
    runs = await list_runs(limit=200)
    cancelled = 0
    for run in runs:
        if run.campaign_id != campaign_id:
            continue
        if run.state in ('finished', 'failed', 'cancelled', 'skipped'):
            continue
        await write_cancel(run.job_id)
        cancelled += 1
    logger.info('curation_train_campaign_cancelled', campaign_id=campaign_id, n=cancelled)
    return cancelled
