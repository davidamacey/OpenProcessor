"""Curation router sub-module: the per-project ingest policy.

``GET/PUT /ingest/policy`` read and replace the project's detect filter and
embedding policy (see :mod:`src.services.curation.ingest_policy`);
``POST /ingest/policy/preview`` counts what a candidate policy would embed
over the items already stored, without writing anything.
"""

from __future__ import annotations

import dataclasses
from collections import Counter

from src.config import get_curation_config
from src.routers.curation._common import OpenSearchDep, _ensure_indexes, get_class_registry, router
from src.routers.curation._config_common_models import ApiErrorResponse, api_error
from src.routers.curation._error_models import (
    DetectorNotServableResponse,
    DetectorUnavailableResponse,
)
from src.routers.curation._ingest_policy_models import (
    IngestPolicyPreview,
    IngestPolicyPutResponse,
    IngestPolicyUpdate,
    PolicyPreviewClass,
)
from src.routers.curation.ingest import _get_detection_profile
from src.services.curation.detector_vocabulary import detector_labels
from src.services.curation.embedding_state import EMBEDDED
from src.services.curation.ingest_detector import detector_problems, effective_profile
from src.services.curation.ingest_policy import (
    DetectorOverride,
    IngestPolicy,
    IngestPolicyBody,
    candidate_from_doc,
    embedding_states,
    unknown_names,
)
from src.services.curation.ingest_policy_store import (
    PolicyConflictError,
    get_ingest_policy,
    put_ingest_policy,
)
from src.services.curation.reprocess_targets import scan_items
from src.utils.class_names import normalize_class_name


PREVIEW_MAX_ITEMS = 100_000
_PREVIEW_FIELDS = [
    'image_id',
    'confidence',
    'crop_area_norm',
    'class_name',
    'proposal_name',
    'class_validated',
    'class_source',
    'label_source',
    'class_labeled_at',
    'class_excluded',
]


def _known_slugs(policy: IngestPolicyBody) -> set[str]:
    """Names a policy can sensibly mention: registry classes and the labels of the
    detector the project will run."""
    known = {c.class_name for c in get_class_registry().load().classes}
    try:
        profile = effective_profile(_get_detection_profile(), policy.detector)
        if profile.detector_model:
            known |= {label.slug for label in detector_labels(profile)}
    except (ValueError, OSError):
        pass
    return known


async def _require_servable(override: DetectorOverride) -> None:
    """``422`` unless the project's own detector is loaded on Triton and serves
    the end2end outputs ingest reads (``503`` when Triton cannot be asked)."""
    from src.main import get_async_triton_pool

    try:
        pool = get_async_triton_pool()
    except RuntimeError as exc:
        raise api_error(503, 'detector_unavailable', f'triton unavailable: {exc}') from exc
    problems = await detector_problems(pool, override)
    if problems:
        raise api_error(
            422,
            'detector_not_servable',
            '; '.join(problems),
            reasons=list(problems),
        )


@router.get('/ingest/policy', response_model=IngestPolicy)
async def get_policy(opensearch: OpenSearchDep) -> IngestPolicy:
    """The project's ingest policy; the defaults (store and embed everything)
    when none was ever written."""
    await _ensure_indexes(opensearch)
    return await get_ingest_policy(opensearch)


@router.put(
    '/ingest/policy',
    response_model=IngestPolicyPutResponse,
    responses={
        409: {'model': ApiErrorResponse},
        422: {'model': DetectorNotServableResponse},
        503: {'model': DetectorUnavailableResponse},
    },
)
async def put_policy(
    body: IngestPolicyUpdate, opensearch: OpenSearchDep
) -> IngestPolicyPutResponse:
    """Replace the policy. ``409`` when ``expected_revision`` is not the stored
    revision (re-read and retry). Affects future ingests and explicit embed
    runs only, never stored data."""
    await _ensure_indexes(opensearch)
    policy_body = IngestPolicyBody(
        detect=body.detect, embedding=body.embedding, detector=body.detector
    )
    if body.detector is not None:
        await _require_servable(body.detector)
    try:
        stored = await put_ingest_policy(
            opensearch, policy_body, expected_revision=body.expected_revision
        )
    except PolicyConflictError as exc:
        raise api_error(409, 'revision_conflict', f'ingest policy changed: {exc}') from exc
    return IngestPolicyPutResponse(
        **stored.model_dump(), unknown_names=unknown_names(policy_body, _known_slugs(policy_body))
    )


@router.post('/ingest/policy/preview', response_model=IngestPolicyPreview)
async def preview_policy(body: IngestPolicyBody, opensearch: OpenSearchDep) -> IngestPolicyPreview:
    """What ``body.embedding`` would embed over the stored items (one image's
    items at a time, the same function ingest runs). Read-only."""
    await _ensure_indexes(opensearch)
    cfg = get_curation_config()
    total = (await opensearch.count(index=cfg.items_index)).get('count', 0)
    docs = await scan_items(
        opensearch,
        {'match_all': {}},
        index=cfg.items_index,
        includes=_PREVIEW_FIELDS,
        max_docs=PREVIEW_MAX_ITEMS,
    )
    by_image: dict[str, list[dict]] = {}
    for _, src in docs:
        by_image.setdefault(src.get('image_id') or '', []).append(src)
    embed: Counter[str] = Counter()
    skip: Counter[str] = Counter()
    display: dict[str, str] = {}
    n_labeled = 0
    for sources in by_image.values():
        cands = [candidate_from_doc(s) for s in sources]
        states = embedding_states(cands, body.embedding)
        unlabeled = embedding_states(
            [dataclasses.replace(c, labeled=False) for c in cands], body.embedding
        )
        for src, state, bare in zip(sources, states, unlabeled, strict=True):
            # Class identity is the NAME (the detector label as stored), never the slug.
            raw = src.get('class_name') or src.get('proposal_name') or ''
            key = normalize_class_name(raw) or 'unnamed'
            name = display.setdefault(key, raw or 'unnamed')
            (embed if state == EMBEDDED else skip)[name] += 1
            n_labeled += state == EMBEDDED and bare != EMBEDDED
    n_embed, n_skip = sum(embed.values()), sum(skip.values())
    names = sorted(set(embed) | set(skip), key=lambda n: -(embed[n] + skip[n]))
    return IngestPolicyPreview(
        total_items=int(total),
        scanned=len(docs),
        truncated=len(docs) >= PREVIEW_MAX_ITEMS and int(total) > len(docs),
        would_embed=n_embed,
        would_not_embed=n_skip,
        embedded_because_labeled=n_labeled,
        estimated_vector_mb=round(n_embed * cfg.encoder_embedding_dim * 4 / 1_000_000, 2),
        by_class=[
            PolicyPreviewClass(name=n, would_embed=embed[n], would_not_embed=skip[n]) for n in names
        ],
    )
