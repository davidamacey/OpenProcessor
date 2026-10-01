"""Every model a user can (or cannot yet) choose, served uniformly in
``GET /config/vocabulary`` ``model_choices`` (W9.8, §7.8.4).

Each row is built from the live source it describes (the Triton index, the
promoted list, the active profile, the ingest env, the endpoint registry,
the curation config's embedding dimensions), never from a constant, so the
UI renders one table and never hardcodes a role. ``settable`` rows say how
to change them (``settable_via``); the rest say why not (``reason``).
"""

from __future__ import annotations

from typing import Any

from src.config import get_curation_config
from src.config.ingest_profiles import ingest_primary_profile, ingest_secondary_profile
from src.config.settings import TritonModelConfig


def _choice(value: str) -> dict[str, str]:
    return {'id': value, 'label': value}


def _row(
    role: str,
    label: str,
    scope: str,
    current: str | None,
    *,
    settable: bool,
    settable_via: str | None = None,
    reason: str | None = None,
    choices: list[str] | None = None,
    dims: int | None = None,
) -> dict[str, Any]:
    return {
        'role': role,
        'label': label,
        'scope': scope,
        'current': current,
        'dims': dims,
        'choices': [_choice(c) for c in (choices or [])],
        'settable': settable,
        'settable_via': settable_via,
        'reason': reason,
    }


def _vlm_rows() -> list[dict[str, Any]]:
    import os

    from src.services.config_store import get_global_config_store
    from src.services.labeling.vlm_catalog import (
        gpu_total_gb_from_env,
        load_catalog,
        local_vlm_status,
    )
    from src.services.labeling.vlm_endpoints import (
        VlmEndpointUnavailableError,
        active_vlm_endpoint,
        available_vlm_endpoints,
        get_vlm_endpoint,
    )

    try:
        active = active_vlm_endpoint()
    except VlmEndpointUnavailableError:
        active = None
    endpoints = [e.name for e in available_vlm_endpoints()]
    local_name = os.environ.get('OP_LOCAL_VLM_ENDPOINT', '').strip()
    local_endpoint = get_vlm_endpoint(local_name) if local_name else None
    probe = local_endpoint.last_probe if local_endpoint else None
    status = local_vlm_status(
        endpoint_name=local_name if local_endpoint else '',
        served_model=local_endpoint.body.model if local_endpoint else None,
        served_root=probe.root if probe else None,
        served_max_model_len=probe.max_model_len if probe else None,
        desired=get_global_config_store().current.local_vlm_desired,
        gpu_total_gb=gpu_total_gb_from_env(os.environ.get('OP_LOCAL_VLM_GPU_TOTAL_MIB')),
    )
    served = status['served'] or {}
    return [
        _row(
            'vlm',
            'VLM endpoint',
            'config_store',
            active.name if active else None,
            settable=True,
            settable_via='POST /vlm/endpoints/{name}/activate, ?vlm= per run',
            choices=endpoints,
        ),
        _row(
            'local_vlm_model',
            'Local VLM model',
            'deployment',
            served.get('catalog_id'),
            settable=bool(status['configured']),
            settable_via='POST /vlm/local/select, then the host command it returns',
            reason=(
                None if status['configured'] else 'This deployment has no in-compose VLM to switch.'
            ),
            choices=[e.id for e in load_catalog()],
        ),
    ]


def build_model_choices(
    *, detector_ids: list[str], ocr_pipeline_ids: list[str], promoted_ids: list[str]
) -> list[dict[str, Any]]:
    """The full table. ``detector_ids`` are the selectable detection models
    (the Triton index plus promoted models), ``ocr_pipeline_ids`` the OCR
    pipeline models, ``promoted_ids`` the promoted (trained) models."""
    from src.services.detection.profile_registry import get_active_region_profile
    from src.services.training.profiles import get_profiles

    config = get_curation_config()
    profile = get_active_region_profile()
    primary = ingest_primary_profile()
    secondary = ingest_secondary_profile()
    sizes = sorted({str(p['model_size']) for p in get_profiles() if p.get('model_size')})
    return [
        _row(
            'detect_model',
            'Detector (per request)',
            'per_request',
            TritonModelConfig.YOLO_MODEL,
            settable=True,
            settable_via='?model_name= on /detect and /detect/batch',
            choices=detector_ids,
        ),
        _row(
            'item_detector',
            'Item detector (ingest)',
            'deployment',
            primary.detector_model or primary.name,
            settable=False,
            settable_via='OP_INGEST_PRIMARY_DETECTOR_MODEL (restart)',
            reason='Ingest detectors are deployment configuration; changing one needs a restart.',
        ),
        _row(
            'secondary_classifier',
            'Secondary classifier (ingest)',
            'deployment',
            (secondary.detector_model or secondary.name) if secondary is not None else None,
            settable=False,
            settable_via='OP_INGEST_SECONDARY_DETECTOR_MODEL (restart)',
            reason='Ingest detectors are deployment configuration; changing one needs a restart.',
        ),
        _row(
            'region_detector',
            'Region detector',
            'region_profile',
            profile.detector_model if profile is not None else None,
            settable=True,
            settable_via='PUT /region_profiles/{name} detector_model',
            choices=detector_ids,
        ),
        _row(
            'region_segmenter',
            'Region segmenter',
            'deployment',
            profile.segmenter_name if profile is not None else None,
            settable=False,
            settable_via='OP_SEGMENTER_URL / OP_SEGMENTER_URLS (the prompt is per profile)',
            reason='One segmenter service backs the deployment; its prompt is set per profile.',
        ),
        _row(
            'region_ocr',
            'Region text (OCR)',
            'region_profile',
            profile.ocr_pipeline_model if profile is not None else None,
            settable=True,
            settable_via='PUT /region_profiles/{name} ocr_pipeline_model / ocr_det_model / ocr_rec_model',
            choices=ocr_pipeline_ids,
        ),
        *_vlm_rows(),
        _row(
            'item_embedding',
            'Item embeddings',
            'deployment',
            TritonModelConfig.PE_IMAGE_MODEL,
            settable=False,
            dims=config.encoder_embedding_dim,
            reason=(
                'Changing it changes vector dimensions and needs a full re-index and '
                're-cluster (future work).'
            ),
        ),
        _row(
            'search_embedding',
            'Search embeddings',
            'deployment',
            TritonModelConfig.CLIP_IMAGE_MODEL,
            settable=False,
            dims=config.embedding_dim,
            reason=(
                'Changing it changes vector dimensions and needs a full re-index (future work).'
            ),
        ),
        _row(
            'face_models',
            'Face models',
            'deployment',
            f'{TritonModelConfig.FACE_DETECT_MODEL} + {TritonModelConfig.ARCFACE_MODEL}',
            settable=False,
            reason='Faces run a fixed detection + recognition pipeline.',
        ),
        _row(
            'training_size',
            'Training model size',
            'per_run',
            None,
            settable=True,
            settable_via='POST /train/start model_size; GET /train/profiles',
            choices=sizes,
        ),
        _row(
            'bakeoff_models',
            'Bake-off models',
            'per_run',
            None,
            settable=True,
            settable_via='POST /bakeoff/run models[]',
            choices=promoted_ids,
        ),
    ]


__all__ = ['build_model_choices']
