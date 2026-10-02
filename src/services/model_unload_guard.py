"""Which models may be unloaded or deleted, and when ``force`` is needed.

One guard for every route that removes a model from Triton: the
project-scoped ``DELETE /models/{name}`` and the public
``DELETE /models/{name}`` (which deletes a whole exported model family).
Unloading the wrong model breaks live serving instantly, so:

- an external-service entry (the segmenter, a VLM endpoint) is not a Triton
  model at all: 400, always;
- the models the configured pipeline depends on (the primary item proposer,
  the optional secondary classifier, the region detector, OCR det/rec): 403,
  never unloadable here, not even with ``force``;
- the other core models (CLIP image encoder, face detect / recognition):
  409 unless ``force``.

Which models those are is driven by the active ingest and region profiles,
never by a hardcoded model list.
"""

from __future__ import annotations

import contextlib

from src.config.ingest_profiles import ingest_primary_profile, ingest_secondary_profile
from src.config.settings import TritonModelConfig
from src.services.detection.profile_registry import get_active_region_profile
from src.services.labeling.vlm_endpoints import available_vlm_endpoints


REGION_PROTECTED_PREFIXES = ('paddleocr_',)


class UnloadRefusedError(Exception):
    """The guard refused the unload; ``status_code`` / ``detail`` are what the
    route answers."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def region_protected_models() -> frozenset[str]:
    primary = ingest_primary_profile()
    secondary = ingest_secondary_profile()
    region = get_active_region_profile()
    region_models = (
        (region.detector_model, region.ocr_det_model, region.ocr_rec_model)
        if region is not None
        else ()
    )
    return frozenset(
        name
        for name in (
            primary.detector_model,
            secondary.detector_model if secondary is not None else None,
            *region_models,
            TritonModelConfig.OCR_DET_MODEL,
            TritonModelConfig.OCR_REC_MODEL,
        )
        if name
    )


def is_region_protected_model(model_name: str) -> bool:
    return model_name in region_protected_models() or model_name.startswith(
        REGION_PROTECTED_PREFIXES
    )


def core_pipeline_models() -> frozenset[str]:
    """Non-region models serving live traffic: unloading one needs ``force``
    (a deliberate re-promote is legitimate, if rare)."""
    return frozenset(
        {
            TritonModelConfig.CLIP_IMAGE_MODEL,
            TritonModelConfig.FACE_DETECT_MODEL,
            TritonModelConfig.ARCFACE_MODEL,
        }
    )


def external_service_model_names() -> frozenset[str]:
    """Names of roster entries that are their own HTTP services, not Triton
    models: the segmenter, and every VLM endpoint's name and model id."""
    names: set[str] = set()
    region = get_active_region_profile()
    if region is not None and region.segmenter_name:
        names.add(region.segmenter_name)
    with contextlib.suppress(Exception):  # an unresolvable registry just skips the VLMs
        for endpoint in available_vlm_endpoints():
            names.update((endpoint.name, endpoint.body.model, endpoint.model_id))
    return frozenset(names)


def check_unload(model_name: str, *, force: bool) -> bool:
    """Raise :class:`UnloadRefusedError` unless ``model_name`` may be unloaded.
    Returns whether it is a core pipeline model (only reachable with
    ``force``), so the caller can warn that live inference is now down."""
    if model_name in external_service_model_names():
        raise UnloadRefusedError(
            400,
            f'{model_name!r} is an external-service model (segmenter or VLM), '
            'not a Triton model. It has no Triton repository entry to unload; '
            'manage it out of band.',
        )
    if is_region_protected_model(model_name):
        raise UnloadRefusedError(
            403,
            f'{model_name!r} is a region-detector pipeline model (detector or OCR) '
            'and can never be unloaded through this endpoint, even with '
            'force=true. Manage it out of band.',
        )
    is_core = model_name in core_pipeline_models()
    if is_core and not force:
        raise UnloadRefusedError(
            409,
            f'{model_name!r} is a core pipeline model currently serving live '
            'traffic. Pass force=true to unload it anyway — this WILL break '
            'live inference for this model until a replacement is loaded.',
        )
    return is_core


__all__ = [
    'REGION_PROTECTED_PREFIXES',
    'UnloadRefusedError',
    'check_unload',
    'core_pipeline_models',
    'external_service_model_names',
    'is_region_protected_model',
    'region_protected_models',
]
