"""Cross-job policy: campaign auto-skip / auto-promote, and the quant hand-off.

A campaign is a set of ``job.json`` files sharing a ``campaign_id``, written in
one go by ``POST {api_prefix}/train/start_campaign``. Nothing coordinates them
at runtime except this module, which runs once per job *after* its terminal
status is on disk (so siblings always see a consistent picture):

* a finished run that cleared its ``stop_when`` thresholds marks every
  still-queued sibling ``skipped``, so the watcher never starts them;
* the last run in a campaign optionally promotes the best sibling by calling
  the API's own promote endpoint.

:func:`write_quant_bakeoff_job` is the other cross-process hand-off: an opt-in
job file for the bake-off evaluator.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from job_protocol import (
    ACTIVE_STATES,
    DEFAULT_EXPORT_BATCH,
    DEFAULT_INPUT_SIZE,
    TERMINAL_STATES,
    _atomic_write_json,
    _job_id_from_path,
    _read_json,
    _status_path_for,
    _utcnow_iso,
)
from logutil import get_logger


if TYPE_CHECKING:
    from job_protocol import JobSpec, StatusState


logger = get_logger('trainer.campaign')


# Bake-off jobs directory, for the opt-in auto-quantize hand-off. Same default
# as docker/evaluator/Dockerfile's OP_BAKEOFF_JOBS_DIR.
BAKEOFF_JOBS_DIR = Path(
    os.environ.get('OP_BAKEOFF_JOBS_DIR', '/var/lib/openprocessor/bakeoff_jobs')
)
BAKEOFF_OUT_DIR = Path(os.environ.get('OP_BAKEOFF_OUT_DIR', '/var/lib/openprocessor/bakeoff_out'))

# API base + prefix for the campaign auto-promote callback. The trainer and the
# API share a compose network, so the service name resolves via Docker DNS.
API_BASE_URL = os.environ.get('OP_API_BASE_URL', 'http://yolo-api:8000')
API_PREFIX = os.environ.get('OP_API_PREFIX', '/curation')


# ---------------------------------------------------------------------------
# Campaign auto-skip + auto-promote
# ---------------------------------------------------------------------------


def _scan_campaign_jobs(jobs_dir: Path, campaign_id: str) -> list[tuple[Path, dict[str, Any]]]:
    """Return ``(job_path, status_or_empty)`` for every sibling in a campaign.

    The status dict is empty when the trainer hasn't written one yet (the run
    is still strictly queued).
    """
    out: list[tuple[Path, dict[str, Any]]] = []
    if not jobs_dir.is_dir():
        return out
    for jp in sorted(jobs_dir.glob('*.job.json')):
        try:
            spec_raw = _read_json(jp)
        except (OSError, ValueError):
            continue  # a corrupt sibling must not stop the scan
        if spec_raw.get('campaign_id') != campaign_id:
            continue
        status_path = _status_path_for(jp)
        status_raw: dict[str, Any] = {}
        if status_path.is_file():
            try:
                status_raw = _read_json(status_path)
            except (OSError, ValueError):
                status_raw = {}
        out.append((jp, status_raw))
    return out


def _job_state(status_raw: dict[str, Any]) -> str:
    """Effective state of a sibling job; no status file means ``queued``."""
    if not status_raw:
        return 'queued'
    return str(status_raw.get('state') or 'queued')


def _write_skipped_status(job_path: Path, spec_raw: dict[str, Any], reason: str) -> None:
    """Write a terminal ``state='skipped'`` status so nothing runs this job."""
    status_path = _status_path_for(job_path)
    payload = {
        'job_id': spec_raw.get('job_id') or _job_id_from_path(job_path),
        'campaign_id': spec_raw.get('campaign_id'),
        'state': 'skipped',
        'finished_at': _utcnow_iso(),
        'heartbeat_at': _utcnow_iso(),
        'error': reason,
    }
    try:
        _atomic_write_json(status_path, payload)
    except OSError as exc:
        logger.warning(
            'campaign: failed to write skipped status', path=str(status_path), error=str(exc)
        )


def stop_when_satisfied(stop_when: dict[str, float] | None, eval_block: dict[str, Any]) -> bool:
    """True when the just-finished run cleared every ``stop_when`` threshold.

    Unknown rules fail *closed* -- we never skip siblings on a policy the
    trainer doesn't understand.
    """
    if not stop_when:
        return False
    keys = {'map50_at_least': 'map50', 'map50_95_at_least': 'map50_95'}
    for key, threshold in stop_when.items():
        try:
            t = float(threshold)
        except (TypeError, ValueError):
            continue
        metric_key = keys.get(key)
        if metric_key is None:
            logger.warning('campaign: unknown stop_when key', key=key)
            return False
        v = eval_block.get(metric_key)
        if v is None or float(v) < t:
            return False
    return True


def _post_promote(job_id: str, triton_name: str) -> bool:
    """POST ``{api_prefix}/train/promote/{job_id}``; return success.

    Errors are logged and swallowed so a flaky API doesn't cascade-fail the
    campaign -- the run itself already finished and its artifacts are on disk.
    """
    try:
        import requests
    except ImportError:
        logger.warning('campaign: requests unavailable; cannot auto-promote')
        return False
    url = f'{API_BASE_URL.rstrip("/")}{API_PREFIX}/train/promote/{job_id}'
    payload = {
        'triton_name': triton_name,
        # Conservative defaults matching the ONNX export below; an operator can
        # re-promote manually with different settings.
        'max_batch_size': DEFAULT_EXPORT_BATCH,
        'input_size': DEFAULT_INPUT_SIZE,
        'fp16': True,
        'overwrite': True,
    }
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
    except Exception as exc:  # any transport failure is non-fatal
        logger.warning('campaign: auto-promote POST failed', url=url, error=str(exc))
        return False
    return True


def maybe_handle_campaign(spec: JobSpec, state: StatusState) -> None:
    """Cross-job campaign actions: auto-skip + auto-promote.

    Called once per job after the terminal status is on disk (so siblings see a
    consistent picture). No-ops when the job isn't part of a campaign.

    1. ``finished`` and ``stop_when`` satisfied -> mark every still-``queued``
       sibling ``skipped`` so the watcher won't pick them up.
    2. This is the last run (``is_last_in_campaign``, or no active siblings
       remain) and ``auto_promote_best`` was requested -> promote the sibling
       with the highest ``eval.map50``.
    """
    campaign_id = spec.campaign_id
    if not campaign_id or state.state not in TERMINAL_STATES:
        return

    jobs_dir = spec.job_path.parent
    siblings = _scan_campaign_jobs(jobs_dir, campaign_id)

    if state.state == 'finished' and stop_when_satisfied(spec.stop_when, state.eval or {}):
        skipped = 0
        for sib_path, sib_status in siblings:
            if sib_path == spec.job_path or _job_state(sib_status) != 'queued':
                continue
            try:
                sib_spec = _read_json(sib_path)
            except (OSError, ValueError):
                continue
            _write_skipped_status(sib_path, sib_spec, reason='auto_skip_threshold_met')
            skipped += 1
        if skipped:
            logger.info(
                'campaign: auto-skipped sibling runs',
                campaign_id=campaign_id,
                triggered_by=spec.job_id,
                skipped=skipped,
            )
        siblings = _scan_campaign_jobs(jobs_dir, campaign_id)

    active_remaining = [
        sp for sp, st in siblings if sp != spec.job_path and _job_state(st) in ACTIVE_STATES
    ]
    if not (spec.is_last_in_campaign or not active_remaining):
        return
    if not spec.auto_promote_best:
        return

    best_job_id: str | None = None
    best_map50 = float('-inf')
    for sib_path, sib_status in siblings:
        if _job_state(sib_status) != 'finished':
            continue
        map50 = (sib_status.get('eval') or {}).get('map50')
        if map50 is None:
            continue
        try:
            value = float(map50)
        except (TypeError, ValueError):
            continue
        if value > best_map50:
            best_map50 = value
            best_job_id = str(sib_status.get('job_id') or _job_id_from_path(sib_path))

    if best_job_id is None:
        logger.info(
            'campaign: auto-promote skipped -- no finished run with map50',
            campaign_id=campaign_id,
        )
        return

    triton_name = f'{campaign_id}_best'
    ok = _post_promote(best_job_id, triton_name)
    logger.info(
        'campaign: auto-promote',
        campaign_id=campaign_id,
        best_job_id=best_job_id,
        best_map50=best_map50,
        triton_name=triton_name,
        ok=ok,
    )


# ---------------------------------------------------------------------------
# Opt-in auto-quantize bake-off hand-off
# ---------------------------------------------------------------------------


def write_quant_bakeoff_job(spec: JobSpec, state: StatusState) -> None:
    """Drop a bake-off job that exports + benchmarks this finished run.

    The on-demand evaluator (``docker/evaluator/``,
    ``scripts/curation/bakeoff/bakeoff_runner.py``) watches the bake-off jobs
    dir; a job carrying a ``quantize`` block exports the checkpoint to portable
    ONNX (fp32/fp16/int8) and scores those variants on the frozen split, so a
    quantization panel updates with no manual step. Opt-in per training job via
    ``job.json``'s ``auto_quantize_bakeoff``.
    """
    if not state.checkpoint_path:
        logger.warning('auto-quantize skipped: no checkpoint', job_id=spec.job_id)
        return
    BAKEOFF_JOBS_DIR.mkdir(parents=True, exist_ok=True)
    bakeoff_id = f'{spec.job_id}_quant'
    dataset = str(spec.dataset_export_dir)
    job = {
        'job_id': bakeoff_id,
        'datasets': [{'name': 'curated', 'path': dataset}],
        'verify_frozen': True,
        'out_dir': str(BAKEOFF_OUT_DIR / bakeoff_id),
        # The .pt itself as the FP32 reference row alongside the exported
        # variants.
        'models': [
            {
                'backend': 'ultralytics',
                'weights': state.checkpoint_path,
                'name': 'candidate',
                'imgsz': DEFAULT_INPUT_SIZE,
                'mode': 'full',
                'device': 'cuda',
            }
        ],
        'quantize': {
            'model_id': f'{spec.job_id}_{spec.model_family}{spec.model_size}',
            'checkpoint': state.checkpoint_path,
            'calib_dataset': dataset,
            'formats': ['fp32_onnx', 'fp16_onnx', 'int8_onnx'],
            'n_calib': 1000,
            'out_root': str(BAKEOFF_OUT_DIR / 'quant'),
            # Also run the steady-state throughput sweep (best-effort inside the
            # evaluator). No 'coreml': that export leg is not shipped, and asking
            # for it only records a failed stage in the job status.
            'throughput': True,
        },
    }
    tmp = BAKEOFF_JOBS_DIR / f'.{bakeoff_id}.job.json.tmp'
    tmp.write_text(json.dumps(job, indent=2), encoding='utf-8')
    tmp.rename(BAKEOFF_JOBS_DIR / f'{bakeoff_id}.job.json')
    logger.info('auto-quantize bake-off enqueued', job_id=spec.job_id, bakeoff_id=bakeoff_id)
