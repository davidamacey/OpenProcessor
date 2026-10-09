"""``GET /config/vocabulary`` (W4, any_domain_plan.md §7.4): the enum
choice lists both the prompt-pack and region-profile editors render.

The final §7.4 shape from this wave: ``ocr`` as lists (the configured
value first, ``configured: true``), ``segmenters[]`` with the *active
profile's* own candidate-selection numbers (this repo has no separate
fixed server-side cap -- see the module docstring below), and ``vlm`` as
``{active, endpoints: [<one env entry>]}`` built from ``OP_VLM_URL`` /
``OP_VLM_MODEL``. W9 swaps the ``vlm`` block's source to the endpoint
registry without changing this shape, and adds ``model_choices`
(served here as an always-empty list until then, so the field exists in
the contract without W9's routes).
"""

from __future__ import annotations

import contextlib
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.routers.curation._common import OpenSearchDep, get_class_registry, router
from src.services.curation.reprocess_vocabulary import (
    VocabEntry,
    filter_field_vocabulary,
    job_status_vocabulary,
    lock_reason_vocabulary,
    scope_vocabulary,
)


class Choice(BaseModel):
    id: str
    label: str


#: Where each ``choices_from`` value on a schema row resolves to (§7.3) --
#: the schema builders and ``test_config_vocabulary.py`` both read this,
#: so the two can never drift apart.
CHOICE_SOURCES: dict[str, str] = {
    'detectors': 'detectors',
    'segmenters': 'segmenters',
    'ocr_pipeline_models': 'ocr.pipeline_models',
    'ocr_rec_models': 'ocr.rec_models',
    'registry_classes': 'registry_classes',
    'text_reader_modes': 'text_reader_modes',
}


class DetectorEntry(BaseModel):
    model_config = {'extra': 'allow'}

    name: str
    choice: Choice
    source: str
    state: str | None = None
    ready: bool = False
    versions: list[str] = Field(default_factory=list)
    promoted_at: str | None = None
    job_id: str | None = None


class SegmenterEntry(BaseModel):
    name: str
    choice: Choice
    endpoint: str
    status: str
    masks: bool = True
    max_candidates: int | None = None
    default_min_score: float | None = None


class VlmEndpointEntry(BaseModel):
    name: str
    source: str
    model: str
    resolved_model: str | None = None
    locality: str
    sends_images_externally: bool
    warning: str | None = None
    # The endpoint's own health (`ready | unprobed | probe_failed |
    # unreachable`, labels in `GET /vlm/endpoints`).
    status: str
    max_images_per_call: int | None = None
    active: bool


class VlmActiveRef(BaseModel):
    name: str | None
    revision: int | None = None


class VlmBlock(BaseModel):
    active: VlmActiveRef
    endpoints: list[VlmEndpointEntry]


class ModelChoice(BaseModel):
    """One row of the "every model choice" table (W9.8, §7.8.4)."""

    role: str
    label: str
    scope: Literal['per_request', 'per_run', 'region_profile', 'config_store', 'deployment']
    current: str | None
    dims: int | None = None
    choices: list[Choice]
    settable: bool
    settable_via: str | None = None
    reason: str | None = None


class OcrModelEntry(BaseModel):
    name: str
    choice: Choice
    source: str
    state: str | None = None
    ready: bool = False
    configured: bool = False


class OcrBlock(BaseModel):
    available: bool
    pipeline_models: list[OcrModelEntry]
    det_models: list[OcrModelEntry]
    rec_models: list[OcrModelEntry]


class TextReaderModeEntry(BaseModel):
    id: str
    choice: Choice
    label: str
    reads_text: bool
    needs_vlm: bool
    needs_ocr: bool


class RegistryClassEntry(BaseModel):
    class_id: int
    class_name: str
    choice: Choice


class PromptPackCallEntry(BaseModel):
    id: str
    label: str


class VocabularyLabels(BaseModel):
    scope: dict[str, str] = Field(
        default_factory=lambda: {
            'per_request': 'Per request',
            'per_run': 'Per run',
            'region_profile': 'In the region profile',
            'config_store': 'Saved setting (hot switch)',
            'deployment': 'Deployment (restart)',
        }
    )


class ReprocessVocabulary(BaseModel):
    scopes: list[VocabEntry]
    filter_fields: list[VocabEntry]
    job_statuses: list[VocabEntry]
    lock_reasons: list[VocabEntry]


class ConfigVocabularyResponse(BaseModel):
    detectors: list[DetectorEntry]
    segmenters: list[SegmenterEntry]
    vlm: VlmBlock
    ocr: OcrBlock
    model_choices: list[ModelChoice] = Field(default_factory=list)
    text_reader_modes: list[TextReaderModeEntry]
    registry_classes: list[RegistryClassEntry]
    prompt_pack_calls: list[PromptPackCallEntry]
    labels: VocabularyLabels = Field(default_factory=VocabularyLabels)
    reprocess: ReprocessVocabulary


_TEXT_READER_LABELS: dict[str, str] = {
    'none': 'Off (region has no text)',
    'vlm': 'VLM',
    'ocr': 'OCR',
    'vlm_then_ocr': 'VLM, then OCR',
    'both': 'VLM and OCR',
}
_TEXT_READER_NEEDS_VLM: frozenset[str] = frozenset({'vlm', 'vlm_then_ocr', 'both'})
_TEXT_READER_NEEDS_OCR: frozenset[str] = frozenset({'ocr', 'vlm_then_ocr', 'both'})

_PROMPT_PACK_CALL_LABELS: dict[str, str] = {
    'classify': 'Classify',
    'open_classify': 'Classify (open vocabulary)',
    'combined': 'Classify + verify region',
    'combined_batch': 'Classify + verify region (batch)',
    'region_verify': 'Verify region',
    'region_verify_batch': 'Verify region (batch)',
    'region_visible': 'Region visible pre-filter',
}


async def _build_detectors(*, include_other_projects: bool) -> list[DetectorEntry]:
    from src.routers.curation._models_class_mapping import (
        bound_registry,
        discover_foreign_shared_models,
        listing_fields,
    )
    from src.services.training.promoted_models import discover_promoted_models
    from src.services.triton_control import TritonControlService

    registry = bound_registry()
    try:
        repo = await TritonControlService().get_repository_index()
    except Exception:
        repo = []
    state_by_name: dict[str, dict[str, Any]] = {e['name']: e for e in repo if 'name' in e}

    raw: dict[str, dict[str, Any]] = {}
    for name, state in state_by_name.items():
        raw[name] = {
            'name': name,
            'choice': Choice(id=name, label=name),
            'source': 'triton',
            'state': state.get('state'),
            'ready': state.get('state') == 'READY',
            'versions': [state['version']] if state.get('version') else [],
        }
    for promoted in discover_promoted_models():
        name = promoted['name']
        state = state_by_name.get(name, {})
        raw[name] = {
            'name': name,
            'choice': Choice(id=name, label=f'{name} (promoted)'),
            'source': 'promoted',
            'state': state.get('state'),
            'ready': state.get('state') == 'READY',
            'versions': [state['version']] if state.get('version') else [],
            'promoted_at': promoted.get('promoted_at'),
            'job_id': promoted.get('job_id'),
        }
    if include_other_projects:
        for shared in discover_foreign_shared_models():
            name = shared['name']
            if name in raw:
                continue
            state = state_by_name.get(name, {})
            raw[name] = {
                'name': name,
                'choice': Choice(id=name, label=f'{name} (shared by {shared.get("project")})'),
                'source': 'promoted',
                'state': state.get('state'),
                'ready': state.get('state') == 'READY',
                'versions': [state['version']] if state.get('version') else [],
                'promoted_at': shared.get('promoted_at'),
                'job_id': shared.get('job_id'),
            }

    out: list[DetectorEntry] = []
    for name in sorted(raw):
        entry = raw[name]
        entry.update(listing_fields(name, registry))
        out.append(DetectorEntry(**entry))
    return out


async def _build_segmenter() -> list[SegmenterEntry]:
    from src.routers.curation._models_segmenter import build_segmenter_entry
    from src.services.detection.profile_registry import get_active_region_profile

    profile = get_active_region_profile()
    if profile is None or not profile.segmenter_name:
        return []
    built = await build_segmenter_entry(
        profile.segmenter_name, profile.segmenter_name, 'segmenter', 'Promptable segmentation'
    )
    return [
        SegmenterEntry(
            name=built['name'],
            choice=Choice(id=built['name'], label=built['name']),
            endpoint=built['endpoint'],
            status=built['status'],
            # No fixed server-side candidate cap exists in this codebase
            # (select_region_candidates takes max_n from the ACTIVE
            # profile's own max_regions_per_item at call time, W8.4) --
            # served here rather than a fabricated constant.
            max_candidates=profile.max_regions_per_item,
            default_min_score=profile.confidence_floor,
        )
    ]


async def _build_vlm(opensearch: Any) -> VlmBlock:
    """The ``vlm`` block from the endpoint registry (the ``env`` built-in
    among the entries); ``active`` is the BOUND project's."""
    from src.services.config_store import get_config_store
    from src.services.labeling.vlm_endpoints import (
        VlmEndpointUnavailableError,
        active_vlm_endpoint,
        available_vlm_endpoints,
        external_ack_state,
        refresh_vlm_state,
    )
    from src.services.labeling.vlm_url_policy import acompute_locality

    with contextlib.suppress(Exception):
        await refresh_vlm_state(opensearch)
    try:
        active = active_vlm_endpoint()
    except VlmEndpointUnavailableError:
        active = None
    ref = get_config_store().current.active_vlm
    entries: list[VlmEndpointEntry] = []
    for endpoint in available_vlm_endpoints():
        try:
            locality = await acompute_locality(endpoint.body.base_url, cached=True)
        except ValueError:
            locality = 'unknown'
        ack = external_ack_state(
            endpoint, locality, active_ref=None, active_ack_at=None, acked_refs=frozenset()
        )
        entries.append(
            VlmEndpointEntry(
                name=endpoint.name,
                source=endpoint.source,
                model=endpoint.body.model,
                resolved_model=endpoint.model_id,
                locality=locality,
                sends_images_externally=ack.sends_images_externally,
                warning=ack.warning,
                status=endpoint.status,
                max_images_per_call=endpoint.body.max_images_per_call,
                active=active is not None and endpoint.name == active.name,
            )
        )
    revision = ref[1] if isinstance(ref, tuple) else None
    return VlmBlock(
        active=VlmActiveRef(name=active.name if active else None, revision=revision),
        endpoints=entries,
    )


async def _build_ocr() -> OcrBlock:
    from src.config.settings import TritonModelConfig
    from src.services.triton_control import TritonControlService

    try:
        repo = await TritonControlService().get_repository_index()
    except Exception:
        repo = []
    state_by_name = {e['name']: e for e in repo if 'name' in e}

    def _entry(name: str) -> OcrModelEntry:
        state = state_by_name.get(name, {})
        return OcrModelEntry(
            name=name,
            choice=Choice(id=name, label=name),
            source='triton',
            state=state.get('state'),
            ready=state.get('state') == 'READY',
            configured=True,
        )

    return OcrBlock(
        available=bool(repo),
        pipeline_models=[_entry(TritonModelConfig.OCR_PIPELINE_MODEL)]
        if getattr(TritonModelConfig, 'OCR_PIPELINE_MODEL', None)
        else [],
        det_models=[_entry(TritonModelConfig.OCR_DET_MODEL)],
        rec_models=[_entry(TritonModelConfig.OCR_REC_MODEL)],
    )


def _build_text_reader_modes() -> list[TextReaderModeEntry]:
    from src.services.detection.region_text import TEXT_READER_MODES

    return [
        TextReaderModeEntry(
            id=mode,
            choice=Choice(id=mode, label=_TEXT_READER_LABELS.get(mode, mode)),
            label=_TEXT_READER_LABELS.get(mode, mode),
            reads_text=mode != 'none',
            needs_vlm=mode in _TEXT_READER_NEEDS_VLM,
            needs_ocr=mode in _TEXT_READER_NEEDS_OCR,
        )
        for mode in sorted(TEXT_READER_MODES)
    ]


def _build_registry_classes() -> list[RegistryClassEntry]:
    reg = get_class_registry().load()
    return [
        RegistryClassEntry(
            class_id=c.class_id,
            class_name=c.class_name,
            choice=Choice(id=c.class_name, label=c.class_name),
        )
        for c in reg.classes
        if not c.deprecated
    ]


def _build_prompt_pack_calls() -> list[PromptPackCallEntry]:
    from src.services.labeling.vlm_prompts import REPLY_KEY_CONTRACT

    return [
        PromptPackCallEntry(id=call_id, label=_PROMPT_PACK_CALL_LABELS.get(call_id, call_id))
        for call_id in REPLY_KEY_CONTRACT
    ]


def _build_model_choices(detectors: list[DetectorEntry], ocr: OcrBlock) -> list[ModelChoice]:
    from src.services.curation.model_choices import build_model_choices

    return [
        ModelChoice.model_validate(row)
        for row in build_model_choices(
            detector_ids=[d.name for d in detectors if d.ready],
            ocr_pipeline_ids=[m.name for m in ocr.pipeline_models],
            promoted_ids=[d.name for d in detectors if d.source == 'promoted'],
        )
    ]


@router.get('/config/vocabulary', response_model=ConfigVocabularyResponse)
async def get_config_vocabulary(
    opensearch: OpenSearchDep, include_other_projects: bool = False
) -> ConfigVocabularyResponse:
    detectors = await _build_detectors(include_other_projects=include_other_projects)
    ocr = await _build_ocr()
    return ConfigVocabularyResponse(
        detectors=detectors,
        segmenters=await _build_segmenter(),
        vlm=await _build_vlm(opensearch),
        ocr=ocr,
        model_choices=_build_model_choices(detectors, ocr),
        text_reader_modes=_build_text_reader_modes(),
        registry_classes=_build_registry_classes(),
        prompt_pack_calls=_build_prompt_pack_calls(),
        reprocess=ReprocessVocabulary(
            scopes=scope_vocabulary(),
            filter_fields=filter_field_vocabulary(),
            job_statuses=job_status_vocabulary(),
            lock_reasons=lock_reason_vocabulary(),
        ),
    )
