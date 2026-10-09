"""Promoted-model discovery and cross-project ownership (moved from
``src.routers.curation.models``, W4 any_domain_plan.md §4.3): so
``profile_validation.py`` (a service module) can check a region
profile's ``detector_model`` against promoted models without importing a
router module. ``src.routers.curation.models`` imports these under their
real names -- no back-compat re-export of the old private names (this
project's no-shims rule, owner decision 2026-09-26).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from src.clients.pe_encoder import PE_IMAGE_MODEL
from src.config.curation import get_curation_config
from src.config.ingest_profiles import ingest_primary_profile, ingest_secondary_profile
from src.config.settings import TritonModelConfig
from src.core.logging import get_logger
from src.services.training.model_classes import is_model_shared, model_owner_project
from src.services.training.triton_repo import resolve_triton_models_dir


if TYPE_CHECKING:
    from pathlib import Path


logger = get_logger(__name__)


def _core_models() -> tuple[tuple[str, str, str, str], ...]:
    """Fixed pipeline-model roster, derived from config rather than
    a hardcoded, domain-specific model list.

    Skips the region-detector / OCR entries entirely when the active
    ``DetectionProfile`` leaves them unset (empty string default) — a
    deployment that hasn't wired a detection profile yet just sees the
    always-present CLIP + PE encoder entries.

    Moved here (W3/W4 review 2026-09-28, Minor 6) from
    ``src.routers.curation.models`` -- :func:`discover_promoted_models`
    below needed it and previously imported it back from that router,
    the exact "service depends on the router it was moved out of" the
    W4 move was supposed to avoid.
    """
    from src.services.detection.profile_registry import get_active_region_profile

    entries: list[tuple[str, str, str, str]] = []
    # The primary item proposer and (if configured) the secondary
    # classifier drive most of the label provenance the labeler shows on
    # /review and /classes (class_source ending '_proposal' / '_model')
    # -- they were missing here entirely, so /models showed nothing for
    # the models that produced most of the labels. Both are resolved
    # from OP_INGEST_PRIMARY_*/OP_INGEST_SECONDARY_* (ingest_profiles.py),
    # never hardcoded.
    primary = ingest_primary_profile()
    if primary.detector_model:
        entries.append(
            (
                primary.detector_model,
                'Primary Item Proposer',
                'Proposes item boxes when images are ingested.',
                'TensorRT detection',
            )
        )
    secondary = ingest_secondary_profile()
    if secondary is not None and secondary.detector_model:
        entries.append(
            (
                secondary.detector_model,
                'Secondary Classifier',
                'Classifies proposed item boxes.',
                'TensorRT classification',
            )
        )
    region = get_active_region_profile()
    if region is not None and region.detector_model:
        entries.append(
            (
                region.detector_model,
                'Region Detector',
                'Finds the region of interest inside each item crop.',
                'TensorRT detection',
            )
        )
    if region is not None and region.segmenter_name:
        entries.append(
            (
                region.segmenter_name,
                'Segmenter',
                'Refines or re-detects the region of interest on crops the '
                'primary detector missed.',
                'Promptable segmentation',
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


def project_owns_model(model_name: str) -> bool:
    """True if ``model_name`` (a ``triton_name``) belongs to the bound
    project's namespace (projects_plan.md §5.3/§5.5: ``triton_name =
    model_prefix + requested``).

    ``default``'s ``model_prefix`` stays the empty string (the one
    deliberate exception to "no default special case" -- every
    pre-projects / core-pipeline model, never namespaced, keeps
    resolving as default's own). A non-empty prefix owns exactly the
    names it produces. A namespaced name (contains ``'__'``) that isn't
    ours belongs to some other project -- registered or not, since only
    a project's own non-empty prefix ever produces one. An *unprefixed*
    name (no ``'__'`` at all) carries no project's namespace, so it is
    owned by ``default`` alone.
    """
    from src.config.projects import DEFAULT_SLUG
    from src.services.projects.registry import get_project_registry

    cfg = get_curation_config()
    # promote.json names the owner outright; the prefix rule alone would
    # hand `default` another project's model once that project is missing
    # from the registry snapshot (deleted, or a stale snapshot).
    recorded_owner = model_owner_project(model_name)
    if recorded_owner is not None and recorded_owner != cfg.project_slug:
        return False
    own_prefix = cfg.model_prefix
    if own_prefix:
        return model_name.startswith(own_prefix)
    other_prefixes = (
        record.resources.model_prefix
        for slug, record in get_project_registry().snapshot().items()
        if slug != DEFAULT_SLUG and record.resources.model_prefix
    )
    return not any(model_name.startswith(prefix) for prefix in other_prefixes)


def discover_promoted_models(models_dir: Path | None = None) -> list[dict[str, Any]]:
    """Models promoted through this pipeline that aren't one of the
    fixed core models.

    Every ``TritonPromoter.promote()`` call writes a ``promote.json``
    back-pointer into the model's repo directory -- its presence is
    exactly "this was promoted through `/curation/train/promote`",
    independent of Triton's own load state.

    Best-effort: any I/O error scanning the repo returns an empty list
    rather than failing the whole caller's response -- this is
    supplementary discovery.
    """
    resolved_dir = models_dir if models_dir is not None else resolve_triton_models_dir()
    fixed_names = {name for name, *_ in _core_models()}
    out: list[dict[str, Any]] = []
    try:
        entries = sorted(resolved_dir.iterdir())
    except OSError:
        return out
    for entry in entries:
        if not entry.is_dir() or entry.name in fixed_names:
            continue
        # Project scoping (projects_plan.md §5.3, D1): the shared Triton
        # repo holds every project's promoted models side by side. Only
        # this project's own are listed here; another project's shared
        # ones come from discover_foreign_shared_models, on request only.
        if not project_owns_model(entry.name):
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


def is_promoted(triton_name: str) -> bool:
    """A promote.json exists for ``triton_name`` at all."""
    return (resolve_triton_models_dir() / triton_name / 'promote.json').is_file()


__all__ = [
    '_core_models',
    'discover_promoted_models',
    'is_model_shared',  # re-exported from model_classes for convenience
    'is_promoted',
    'model_owner_project',  # re-exported from model_classes for convenience
    'project_owns_model',
]
