"""Curation /health + /models/status + /models/{name} (unload) endpoints.

The reference implementation this was ported from hardcodes a fixed
vehicle/license-plate model roster (a COCO region proposer, an 80-class
vehicle classifier, an LPR detector, …) with vehicle-domain friendly
names. This port drives the equivalent roster off the already-generic
config this plan built: :class:`~src.config.detection_profile.DetectionProfile`
(the configured region detector + OCR det/rec models) and
:class:`~src.config.settings.TritonModelConfig` (CLIP/face-recognition
models already generic on ``origin/main``), plus the PE-Core image
encoder. A deployment with a different ``DetectionProfile`` gets its own
roster and its own "never unload this" guard for free.
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, Any, Literal

import httpx
from fastapi import HTTPException, Query
from pydantic import BaseModel

from src.clients.pe_encoder import PE_IMAGE_MODEL
from src.config.settings import TritonModelConfig
from src.core.dependencies import AsyncTritonDep  # noqa: TC001
from src.routers.curation._common import (
    CURATION_CLASSES_INDEX,
    CURATION_IMAGES_INDEX,
    CURATION_ITEMS_INDEX,
    CURATION_LABELS_CONFIRMED_INDEX,
    HealthResponse,
    OpenSearchDep,
    get_class_registry,
    logger,
    router,
)
from src.routers.curation.vlm import _get_vlm_labeler
from src.services.detection.profile_registry import get_active_region_profile
from src.services.training.triton_promote import (
    DEFAULT_TRITON_MODELS_DIR,
    ModelNotPromotedError,
    PromoteError,
    UnloadResult,
    unload_triton_model,
)


if TYPE_CHECKING:
    from pathlib import Path


@router.get('/health', response_model=HealthResponse)
async def curation_health(
    opensearch: OpenSearchDep,
    triton_pool: AsyncTritonDep,
) -> HealthResponse:
    """Aggregated health: Triton + OpenSearch + VLM + registry mtime."""
    triton_status: dict[str, Any] = {'reachable': False, 'detail': ''}
    try:
        # AsyncTritonDep is a single AsyncInferenceServerClient, not the pool.
        # Both have is_server_live; fall back to is_server_ready.
        if hasattr(triton_pool, 'is_server_live'):
            ok = await triton_pool.is_server_live()
        elif hasattr(triton_pool, 'health_check'):
            ok = await triton_pool.health_check()
        else:
            ok = False
        triton_status['reachable'] = bool(ok)
    except Exception as exc:
        triton_status['detail'] = str(exc)

    # opensearch dep here is the project's wrapper. Reach the raw async client
    # via attributes commonly exposed; fall back to assuming `opensearch` IS
    # an AsyncOpenSearch.
    raw_os = getattr(opensearch, 'client', None) or opensearch

    os_status: dict[str, Any] = {'reachable': False, 'indexes': {}}
    try:
        for idx_name in (
            CURATION_IMAGES_INDEX,
            CURATION_ITEMS_INDEX,
            CURATION_LABELS_CONFIRMED_INDEX,
            CURATION_CLASSES_INDEX,
        ):
            os_status['indexes'][idx_name] = bool(await raw_os.indices.exists(index=idx_name))
        os_status['reachable'] = True
    except Exception as exc:
        os_status['detail'] = str(exc)

    vlm_status: dict[str, Any] = {'reachable': False}
    try:
        labeler = _get_vlm_labeler()
        h = await labeler.health()
        vlm_status['reachable'] = h.reachable
        vlm_status['model'] = h.model
        if h.last_error:
            vlm_status['last_error'] = h.last_error
    except Exception as exc:
        vlm_status['detail'] = str(exc)

    reg_path = get_class_registry().path
    registry_status: dict[str, Any] = {
        'path': str(reg_path),
        'exists': reg_path.exists(),
    }
    if reg_path.exists():
        try:
            registry_status['mtime'] = datetime.fromtimestamp(
                reg_path.stat().st_mtime, tz=UTC
            ).isoformat()
        except OSError as exc:
            registry_status['detail'] = str(exc)

    overall: Literal['ok', 'degraded', 'down']
    if triton_status['reachable'] and os_status['reachable'] and registry_status.get('exists'):
        overall = 'ok' if vlm_status['reachable'] else 'degraded'
    elif os_status['reachable']:
        overall = 'degraded'
    else:
        overall = 'down'

    return HealthResponse(
        status=overall,
        triton=triton_status,
        opensearch=os_status,
        vlm=vlm_status,
        registry=registry_status,
    )


def _core_models() -> tuple[tuple[str, str, str, str], ...]:
    """Fixed pipeline-model roster, derived from config rather than
    hardcoded vehicle-domain names.

    Skips the region-detector / OCR entries entirely when the active
    ``DetectionProfile`` leaves them unset (empty string default) — a
    deployment that hasn't wired a detection profile yet just sees the
    always-present CLIP + PE encoder entries.
    """
    entries: list[tuple[str, str, str, str]] = []
    region = get_active_region_profile()
    if region is not None and region.detector_model:
        entries.append(
            (
                region.detector_model,
                'Region Detector',
                'Finds the configured region-of-interest (see DetectionProfile) '
                'inside each item crop.',
                'TensorRT detection',
            )
        )
    if region is not None and region.ocr_det_model:
        entries.append(
            (
                region.ocr_det_model,
                'OCR Text Detector',
                'Locates text regions inside a crop to seed a re-detection pass.',
                'TensorRT detection',
            )
        )
    if region is not None and region.ocr_rec_model:
        entries.append(
            (
                region.ocr_rec_model,
                'OCR Text Recognizer',
                'Reads text out of a located text region.',
                'TensorRT recognition',
            )
        )
    entries.append(
        (
            TritonModelConfig.CLIP_IMAGE_MODEL,
            'CLIP Image Encoder',
            'Generates image embeddings for visual search and clustering.',
            'TensorRT/ONNX encoder',
        )
    )
    entries.append(
        (
            PE_IMAGE_MODEL,
            'PE-Core-L14-336 Image Encoder',
            'Generates 1024-d unit-norm embeddings for semantic search.',
            'ONNX Runtime encoder',
        )
    )
    return tuple(entries)


# =============================================================================
# Unload guard classification
# =============================================================================
#
# Defined here (ahead of `models_status`) so the `/models/status` payload
# can carry the same `is_region_detector` / `requires_force_to_unload`
# flags the `DELETE /models/{name}` endpoint enforces below — one source
# of truth, so the UI never has to re-derive (and possibly drift from)
# the guard.

# Region-detector + OCR pipeline models — never unloadable through the
# unload endpoint, not even with force=true. "Never touch the configured
# detection pipeline's models" is the standing constraint; which models
# that means is driven by the active DetectionProfile, not a hardcoded
# domain name.
_REGION_PROTECTED_PREFIXES = ('lpr_', 'paddleocr_')


def _region_protected_models() -> frozenset[str]:
    region = get_active_region_profile()
    region_models = (
        (region.detector_model, region.ocr_det_model, region.ocr_rec_model)
        if region is not None
        else ()
    )
    names = {
        name
        for name in (
            *region_models,
            TritonModelConfig.OCR_DET_MODEL,
            TritonModelConfig.OCR_REC_MODEL,
        )
        if name
    }
    return frozenset(names)


def _is_region_protected_model(model_name: str) -> bool:
    return model_name in _region_protected_models() or model_name.startswith(
        _REGION_PROTECTED_PREFIXES
    )


def _core_pipeline_models() -> frozenset[str]:
    """Non-region-detector models currently serving live production traffic.

    Unloading any of these breaks real ingest/search/labeling right now, so
    they require ``force=true`` (same as the region detector) rather
    than being hard-blocked — a deliberate re-promote is a legitimate (if
    rare) operation an operator should still be able to force through the
    unload endpoint.
    """
    return frozenset(
        {
            TritonModelConfig.CLIP_IMAGE_MODEL,
            TritonModelConfig.FACE_DETECT_MODEL,
            TritonModelConfig.ARCFACE_MODEL,
        }
    )


def _discover_promoted_models(
    models_dir: Path = DEFAULT_TRITON_MODELS_DIR,
) -> list[dict[str, Any]]:
    """Models promoted through this pipeline that aren't one of the fixed
    :func:`_core_models`.

    Every ``TritonPromoter.promote()`` call writes a ``promote.json``
    back-pointer into the model's repo directory — its presence is
    exactly "this was promoted through `/curation/train/promote`",
    independent of Triton's own load state. Surfacing these lets
    `/models` show (and the unload endpoint remove) a throwaway/
    experimental promote without a shell into the host.

    Best-effort: any I/O error scanning the repo returns an empty list
    rather than failing the whole `/models/status` response — this is
    supplementary discovery, not the pipeline's core models.
    """
    fixed_names = {name for name, *_ in _core_models()}
    out: list[dict[str, Any]] = []
    try:
        entries = sorted(models_dir.iterdir())
    except OSError:
        return out
    for entry in entries:
        if not entry.is_dir() or entry.name in fixed_names:
            continue
        promote_json = entry / 'promote.json'
        if not promote_json.is_file():
            continue
        try:
            meta = json.loads(promote_json.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            logger.warning('models_status_promote_json_unreadable', name=entry.name, error=str(exc))
            meta = {}
        out.append(
            {
                'name': entry.name,
                'job_id': meta.get('job_id'),
                'version': meta.get('version'),
                'promoted_at': meta.get('promoted_at'),
            }
        )
    return out


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
async def models_status() -> dict[str, Any]:
    """Status + usage stats for the models that drive the curation labeling pipeline.

    Returns a single ``{"models": [...]}`` object describing each Triton model
    the labeler depends on, plus the external VLM service. Each entry
    carries enough metadata for the labeler ``/models`` page to render a
    self-explanatory card without requiring access to Triton/Prometheus directly.
    """
    triton_http = os.environ.get('TRITON_HTTP_URL', 'http://triton-server:8000')
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
    ) -> dict[str, Any]:
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
            # _is_region_protected_model / _core_pipeline_models above).
            'is_region_protected': _is_region_protected_model(name),
            'requires_force_to_unload': name in _core_pipeline_models(),
            'job_id': job_id,
            'promoted_at': promoted_at,
        }

    models: list[dict[str, Any]] = [
        _build_triton_entry(name, friendly, role, mtype)
        for name, friendly, role, mtype in _core_models()
    ]

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
        for promoted in _discover_promoted_models()
    )

    vlm_status: str = 'unavailable'
    vlm_error: str | None = None
    vlm_model_name = 'vlm'
    try:
        labeler = _get_vlm_labeler()
        vlm_model_name = labeler.model
        h = await labeler.health()
        vlm_status = 'ready' if h.reachable else 'unavailable'
        vlm_error = h.last_error
    except Exception as exc:
        vlm_error = str(exc)

    models.append(
        {
            'name': vlm_model_name,
            'friendly_name': f'VLM ({vlm_model_name})',
            'role': 'Open-vocabulary labeling and region verification',
            'kind': 'external',
            'model_type': 'Vision-Language Model (vLLM)',
            'status': vlm_status,
            'version': None,
            'inference_count': None,
            'exec_count': None,
            'inference_failed': None,
            'avg_latency_ms': None,
            'last_error': vlm_error,
            'endpoint': os.environ.get('OPENWEBUI_BASE_URL', 'http://host.docker.internal:8012/v1'),
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
# bad promote at worst fails to load. (`_is_region_protected_model` /
# `_core_pipeline_models` are defined above, alongside `models_status`,
# which surfaces the same flags per-model so the UI doesn't have to
# re-derive them.)


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
    - Region-detector + OCR models (per the active ``DetectionProfile``)
      can **never** be unloaded here, even with ``force=true`` — 403
      unconditionally.
    - Other core pipeline models (CLIP image encoder, face
      detect/recognition) need ``force=true`` — without it, 409 with a
      loud explanation. Unloading any of them breaks live serving until
      something else is loaded.
    """
    if _is_region_protected_model(model_name):
        raise HTTPException(
            status_code=403,
            detail=(
                f'{model_name!r} is a region-detector pipeline model (detector or OCR) '
                'and can never be unloaded through this endpoint, even with '
                'force=true. Manage it out of band.'
            ),
        )

    is_core = model_name in _core_pipeline_models()
    if is_core and not force:
        raise HTTPException(
            status_code=409,
            detail=(
                f'{model_name!r} is a core pipeline model currently serving live '
                'traffic. Pass force=true to unload it anyway — this WILL break '
                'live inference for this model until a replacement is loaded.'
            ),
        )

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
