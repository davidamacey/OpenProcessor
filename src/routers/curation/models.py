"""Curation /models/status + /models/{name} (unload) endpoints (``/health`` lives in ``health.py``).

Drives the pipeline-model roster off generic config rather than any
hardcoded, domain-specific model list:
:class:`~src.config.detection_profile.DetectionProfile` (the configured
region detector + OCR det/rec models) and
:class:`~src.config.settings.TritonModelConfig` (CLIP/face-recognition
models), plus the PE-Core image encoder. A deployment with a different
``DetectionProfile`` gets its own roster and its own "never unload this"
guard for free.
"""

from __future__ import annotations

import os
import re
from typing import Annotated, Any

import httpx
from fastapi import HTTPException, Query
from pydantic import BaseModel

from src.routers.curation._common import logger, router
from src.routers.curation._models_class_mapping import (
    bound_registry,
    discover_foreign_shared_models,
    listing_fields,
)
from src.routers.curation._models_segmenter import build_segmenter_entry
from src.routers.curation._models_vlm import vlm_status_rows
from src.services.detection.profile_registry import get_active_region_profile
from src.services.model_unload_guard import (
    UnloadRefusedError,
    check_unload,
    core_pipeline_models,
    is_region_protected_model,
)
from src.services.training.promoted_models import (
    _core_models,
    discover_promoted_models,
    project_owns_model,
)
from src.services.training.triton_promote import (
    ModelNotPromotedError,
    PromoteError,
    UnloadResult,
    resolve_triton_http_url,
    unload_triton_model,
)


_TRITON_METRIC_KEYS: dict[str, str] = {
    'nv_inference_count': 'inference_count',
    'nv_inference_exec_count': 'exec_count',
    'nv_inference_request_duration_us': 'request_duration_us',
    'nv_inference_compute_infer_duration_us': 'compute_infer_us',
    'nv_inference_request_failure': 'inference_failed',
}

# Matches `nv_inference_count{model="foo",version="1"} 123.0` and similar.
_TRITON_METRIC_LINE_RE = re.compile(
    r'^(?P<key>nv_inference_\w+)\{(?P<labels>[^}]*)\}\s+(?P<value>\S+)$'
)


def _parse_triton_metrics(text: str) -> dict[str, dict[str, float]]:
    """Parse Triton's Prometheus /metrics text into per-model counter sums."""
    out: dict[str, dict[str, float]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        m = _TRITON_METRIC_LINE_RE.match(line)
        if not m:
            continue
        out_key = _TRITON_METRIC_KEYS.get(m.group('key'))
        if out_key is None:
            continue
        labels = dict(re.findall(r'(\w+)="([^"]*)"', m.group('labels')))
        model = labels.get('model')
        if not model:
            continue
        try:
            out.setdefault(model, {})[out_key] = float(m.group('value'))
        except ValueError:
            continue
    return out


@router.get('/models/status')
async def models_status(
    include_other_projects: Annotated[
        bool,
        Query(description="Also list other projects' promoted models their owners shared"),
    ] = False,
) -> dict[str, Any]:
    """Status + usage stats for the models that drive the curation labeling pipeline.

    Returns a single ``{"models": [...]}`` object describing each Triton model
    the labeler depends on, plus the external VLM and segmenter services.
    Each entry carries enough metadata for the labeler ``/models`` page to
    render a self-explanatory card without requiring access to
    Triton/Prometheus directly. Every Triton entry also carries
    ``project``, ``shared`` and ``class_mapping`` (§5.5: its classes
    matched by name onto this project's registry).
    """
    triton_http = resolve_triton_http_url()
    triton_metrics_url = os.environ.get('TRITON_METRICS_URL', 'http://triton-server:8002/metrics')

    state_by_name: dict[str, dict[str, Any]] = {}
    metrics_by_model: dict[str, dict[str, float]] = {}
    triton_reachable = False

    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            r = await client.post(f'{triton_http}/v2/repository/index')
            if r.status_code == 200:
                triton_reachable = True
                for entry in r.json():
                    state_by_name[entry['name']] = entry
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning('models_status_repo_index_failed', error=str(exc))

        try:
            r = await client.get(triton_metrics_url)
            if r.status_code == 200:
                metrics_by_model = _parse_triton_metrics(r.text)
        except httpx.HTTPError as exc:
            logger.warning('models_status_metrics_failed', error=str(exc))

    def _build_triton_entry(
        name: str,
        friendly: str,
        role: str,
        mtype: str,
        *,
        job_id: str | None = None,
        promoted_at: str | None = None,
        optional: bool = False,
    ) -> dict[str, Any]:
        in_index = name in state_by_name
        state = state_by_name.get(name, {})
        ready = state.get('state') == 'READY'
        m = metrics_by_model.get(name, {})
        ic = int(m.get('inference_count', 0))
        compute_us = float(m.get('compute_infer_us', 0.0))
        avg_ms: float | None = (compute_us / ic / 1000.0) if ic > 0 else None
        if not triton_reachable:
            status = 'unavailable'
        elif ready:
            status = 'ready'
        elif optional and not in_index:
            # Only the profile's region detector is ever marked `optional`
            # (when a segmenter is configured as its fallback -- see the
            # `models_status` call site). A model that's simply absent from
            # Triton's repository index entirely (never shipped/installed),
            # as opposed to present-but-unloaded or present-but-failed
            # (which stay `not_ready`, unchanged), isn't a stall when the
            # cascade already falls back to a ready segmenter -- see
            # region_dependency_health.stall_reason for the matching logic.
            status = 'not_installed'
        else:
            status = 'not_ready'
        return {
            'name': name,
            'friendly_name': friendly,
            'role': role,
            'kind': 'triton',
            'model_type': mtype,
            'status': status,
            'version': state.get('version'),
            'inference_count': ic,
            'exec_count': int(m.get('exec_count', 0)),
            'inference_failed': int(m.get('inference_failed', 0)),
            'avg_latency_ms': round(avg_ms, 2) if avg_ms is not None else None,
            'last_error': None,
            'endpoint': triton_http,
            # Unload guard flags — same source of truth the
            # DELETE /models/{name} endpoint enforces (see
            # src/services/model_unload_guard.py).
            'is_region_protected': is_region_protected_model(name),
            'requires_force_to_unload': name in core_pipeline_models(),
            'job_id': job_id,
            'promoted_at': promoted_at,
            # True only for the active region profile's detector when a
            # segmenter is configured as its fallback (mirrors
            # region_dependency_health.stall_reason's "ready segmenter
            # means a down detector isn't a stall" semantics) -- the
            # frontend uses this to render "optional, not installed"
            # instead of a red NOT READY. False for every other entry.
            'optional': optional,
            # A real Triton model repository entry — DELETE /models/{name}
            # can act on it (subject to the guard flags above). Contrast
            # with external-service entries (segmenter, VLM), which have
            # no Triton repository entry at all.
            'unloadable': True,
        }

    region = get_active_region_profile()
    segmenter_name = region.segmenter_name if region is not None else None

    models: list[dict[str, Any]] = []
    for name, friendly, role, mtype in _core_models():
        # The segmenter is its own HTTP service (OP_SEGMENTER_URL), never a
        # Triton model — probing it via the Triton repository index always
        # reported `not_ready` even while healthy. See build_segmenter_entry.
        if segmenter_name and name == segmenter_name:
            models.append(await build_segmenter_entry(name, friendly, role, mtype))
        elif (
            region is not None
            and region.detector_model
            and name == region.detector_model
            and segmenter_name
        ):
            # The cascade falls back to the segmenter when the region
            # detector is missing (see region_dependency_health's
            # stall_reason) -- so a detector that isn't installed in
            # Triton at all is not a red NOT READY here, it's optional.
            models.append(_build_triton_entry(name, friendly, role, mtype, optional=True))
        else:
            models.append(_build_triton_entry(name, friendly, role, mtype))

    # Surface any additional model promoted through this pipeline (carries
    # promote.json) that isn't one of the fixed core models above, so a
    # throwaway/experimental promote is visible (and unloadable) from
    # /models without a shell into the host.
    models.extend(
        _build_triton_entry(
            promoted['name'],
            f'{promoted["name"]} (promoted)',
            'Promoted via /curation/train/promote'
            + (f' from job {promoted["job_id"]}' if promoted.get('job_id') else ''),
            'Promoted checkpoint',
            job_id=promoted.get('job_id'),
            promoted_at=promoted.get('promoted_at'),
        )
        for promoted in discover_promoted_models()
    )
    if include_other_projects:
        models.extend(
            _build_triton_entry(
                shared['name'],
                f'{shared["name"]} (shared by {shared["project"]})',
                f'Promoted in project {shared["project"]} and shared with other projects',
                'Promoted checkpoint',
                job_id=shared.get('job_id'),
                promoted_at=shared.get('promoted_at'),
            )
            for shared in discover_foreign_shared_models()
        )

    models.extend(await vlm_status_rows())

    # External services (segmenter, VLM) belong to no project and have no
    # class list of their own.
    registry = bound_registry()
    for entry in models:
        if entry['kind'] == 'triton':
            entry.update(listing_fields(entry['name'], registry))
        else:
            entry.update(
                {
                    'project': None,
                    'shared': False,
                    'class_mapping': None,
                    'owned': False,
                    'sharing_revision': None,
                }
            )
    return {'models': models}


# =============================================================================
# Unload / remove a promoted model
# =============================================================================
#
# Mirrors src/services/training/triton_promote.py's promote() path (same
# Triton /v2/repository/models/<name>/{load,unload} control endpoint, same
# on-disk model repo) but in reverse, with guardrails a promote doesn't
# need: unloading the wrong model breaks live serving instantly, where a
# bad promote at worst fails to load. (The guard lives in
# src/services/model_unload_guard.py; `models_status` surfaces the same
# flags per-model so the UI doesn't have to re-derive them.)


# PUT /models/{model_name}/sharing lives in _models_sharing.py (kept
# under the 700-LOC ratchet); imported for its route-registration side
# effect.
from src.routers.curation import _models_sharing  # noqa: E402,F401


class UnloadModelResponse(BaseModel):
    triton_name: str
    triton_unloaded: bool
    directory_removed: bool
    forced: bool
    warning: str | None = None


@router.delete('/models/{model_name}', response_model=UnloadModelResponse)
async def unload_model(
    model_name: str,
    force: Annotated[
        bool,
        Query(description='Bypass the region-detector / core-pipeline-model guard'),
    ] = False,
) -> UnloadModelResponse:
    """Unload ``model_name`` from Triton and delete its model repo directory.

    Guardrails:
    - External-service entries (the segmenter, the VLM) are rejected
      unconditionally: there is no Triton repository entry to unload, and
      treating their name as one would just surface a confusing Triton
      404.
    - Region-detector + OCR models (per the active ``DetectionProfile``)
      can **never** be unloaded here, even with ``force=true`` — 403
      unconditionally.
    - Other core pipeline models (CLIP image encoder, face
      detect/recognition) need ``force=true`` — without it, 409 with a
      loud explanation. Unloading any of them breaks live serving until
      something else is loaded.
    """
    if not project_owns_model(model_name):
        raise HTTPException(
            status_code=404,
            detail=f'{model_name!r} is not a model owned by this project',
        )

    try:
        is_core = check_unload(model_name, force=force)
    except UnloadRefusedError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    try:
        result: UnloadResult = await unload_triton_model(model_name)
    except ModelNotPromotedError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PromoteError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    logger.warning(
        'model_unloaded',
        triton_name=model_name,
        forced=force,
        was_core=is_core,
    )
    warning: str | None = None
    if is_core:
        warning = (
            f'{model_name!r} was forcibly unloaded while it was a core pipeline '
            'model — live inference for this model is down until another model '
            'is loaded/promoted.'
        )
    return UnloadModelResponse(
        triton_name=result.triton_name,
        triton_unloaded=result.triton_unloaded,
        directory_removed=result.directory_removed,
        forced=force,
        warning=warning,
    )
