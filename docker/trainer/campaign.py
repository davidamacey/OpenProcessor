"""Cross-job policy: campaign auto-skip / auto-promote, and the quant hand-off.

A campaign is a set of ``train`` jobs sharing ``params.campaign_id``, queued in one go by
``POST {api_prefix}/projects/{project}/train/start_campaign``. Nothing coordinates them at runtime
except this module, which runs once per run, in the handler, after the run's manifest is stored
(so siblings always see a consistent picture):

* a finished run that cleared its ``stop_when`` thresholds cancels every still-queued sibling
  with the reason ``auto_skip_threshold_met`` (the wire calls them ``skipped``);
* the last run in a campaign optionally promotes the best sibling by calling the API's own
  promote endpoint.

:func:`write_quant_bakeoff_job` is the other cross-process hand-off: an opt-in
``POST {api_prefix}/projects/{project}/bakeoff/run`` that benchmarks the finished run.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from job_spec import DEFAULT_EXPORT_BATCH, DEFAULT_INPUT_SIZE
from logutil import get_logger

from src.db.repos import training_runs as runs_repo
from src.domain import job_states as js
from src.domain.training_keys import SKIPPED_ERROR_CODE


if TYPE_CHECKING:
    from job_spec import JobSpec
    from job_status import StatusState

    from src.tasks.runtime.context import JobContext


logger = get_logger('trainer.campaign')


# API base + prefix for the campaign auto-promote and auto-quantize callbacks. The trainer and the
# API share a compose network, so the service name resolves via Docker DNS.
API_BASE_URL = os.environ.get('OP_API_BASE_URL', 'http://api:8000')
API_PREFIX = os.environ.get('OP_API_PREFIX', '/curation')


# ---------------------------------------------------------------------------
# Campaign auto-skip + auto-promote
# ---------------------------------------------------------------------------


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


def project_api_url(project: str | None) -> str | None:
    """``{API_BASE_URL}{API_PREFIX}/projects/{project}``, or ``None`` when
    the job names no project (a call without one would have no route)."""
    if not project:
        return None
    return f'{API_BASE_URL.rstrip("/")}{API_PREFIX}/projects/{project}'


def _post_promote(project: str | None, job_id: str, triton_name: str) -> bool:
    """POST ``{api_prefix}/projects/{project}/train/promote/{job_id}``; return success.

    Errors are logged and swallowed so a flaky API doesn't cascade-fail the
    campaign -- the run itself already finished and its artifacts are stored.
    """
    try:
        import requests
    except ImportError:
        logger.warning('campaign: requests unavailable; cannot auto-promote')
        return False
    base = project_api_url(project)
    if base is None:
        logger.warning('campaign: the run names no project; cannot auto-promote', job_id=job_id)
        return False
    url = f'{base}/train/promote/{job_id}'
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
    except requests.RequestException as exc:  # connection, timeout and HTTP-status errors
        logger.warning('campaign: auto-promote POST failed', url=url, error=str(exc))
        return False
    return True


_TERMINAL_STATES = frozenset({'finished', 'failed', 'cancelled'})


def _map50(eval_block: Any) -> float | None:
    value = eval_block.get('map50') if isinstance(eval_block, dict) else None
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


async def maybe_handle_campaign(ctx: JobContext, spec: JobSpec, state: StatusState) -> None:
    """Cross-job campaign actions: auto-skip + auto-promote.

    Called once per run after its manifest is stored. No-ops when the run isn't part of a
    campaign.

    1. ``finished`` and ``stop_when`` satisfied -> cancel every still-``queued`` sibling so the
       queue won't start them.
    2. This is the last run (``is_last_in_campaign``, or no active siblings remain) and
       ``auto_promote_best`` was requested -> promote the sibling with the highest
       ``eval.map50``.
    """
    campaign_id = spec.campaign_id
    if not campaign_id or state.state not in _TERMINAL_STATES:
        return

    async with ctx.session() as session:
        if state.state == 'finished' and stop_when_satisfied(spec.stop_when, state.eval or {}):
            skipped = await session.run_sync(
                lambda sync: runs_repo.skip_queued(
                    sync, campaign_id, ctx.job_id, SKIPPED_ERROR_CODE, 'stop_when threshold met'
                )
            )
            if skipped:
                logger.info(
                    'campaign: auto-skipped sibling runs',
                    campaign_id=campaign_id,
                    triggered_by=spec.job_id,
                    skipped=skipped,
                )
        siblings = await session.run_sync(lambda sync: runs_repo.in_campaign(sync, campaign_id))

    others = [r for r in siblings if r.job_id != ctx.job_id]
    active_remaining = [r for r in others if r.status in js.ACTIVE]
    if not (spec.is_last_in_campaign or not active_remaining):
        return
    if not spec.auto_promote_best:
        return

    candidates: list[tuple[str, float]] = []
    own = _map50(state.eval) if state.state == 'finished' else None
    if own is not None:
        candidates.append((spec.job_id, own))
    for run in others:
        if run.status not in (js.COMPLETED, js.COMPLETED_WITH_ERRORS):
            continue
        value = _map50(((run.manifest or {}).get('results') or {}).get('eval'))
        if value is not None:
            candidates.append((str(run.job_id), value))
    if not candidates:
        logger.info(
            'campaign: auto-promote skipped -- no finished run with map50', campaign_id=campaign_id
        )
        return

    best_job_id, best_map50 = max(candidates, key=lambda c: c[1])
    triton_name = f'{campaign_id}_best'
    ok = _post_promote(spec.project, best_job_id, triton_name)
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
    """Ask the API to export + benchmark this finished run
    (``POST {api_prefix}/projects/{project}/bakeoff/run``).

    The bake-off job (``scripts/curation/bakeoff/bakeoff_runner.py`` in the
    evaluator container) exports the checkpoint to portable ONNX
    (fp32/fp16/int8) and scores the run and those variants on the test split
    of the export it trained on, so a quantization panel updates with no
    manual step. Going through the API (not a job file) lets it resolve the
    run's checkpoint, imgsz and class mapping and claim the GPU. Opt-in per
    training job via the spec's ``auto_quantize_bakeoff``; called after
    the run manifest is stored.

    A failed POST is logged and swallowed: the training run already finished.
    """
    if not state.checkpoint_path:
        logger.warning('auto-quantize skipped: no checkpoint', job_id=spec.job_id)
        return
    try:
        import requests
    except ImportError:
        logger.warning('auto-quantize skipped: requests unavailable', job_id=spec.job_id)
        return
    base = project_api_url(spec.project)
    if base is None:
        logger.warning('auto-quantize skipped: the run names no project', job_id=spec.job_id)
        return
    url = f'{base}/bakeoff/run'
    payload = {
        'job_id': f'{spec.job_id}_quant',
        'datasets': [{'id': f'run:{spec.job_id}'}],
        'models': [{'source': 'run', 'run_id': spec.job_id}],
        'quantize': {
            'run_id': spec.job_id,
            'formats': ['fp32_onnx', 'fp16_onnx', 'int8_onnx'],
            'n_calib': 1000,
            'throughput': True,
        },
    }
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as exc:  # connection, timeout and HTTP-status errors
        logger.warning(
            'auto-quantize bake-off POST failed', job_id=spec.job_id, url=url, error=str(exc)
        )
        return
    logger.info('auto-quantize bake-off enqueued', job_id=spec.job_id, bakeoff_id=payload['job_id'])
