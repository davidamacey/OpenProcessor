"""Curation router sub-module: seed the class registry from the ingest detector."""

from __future__ import annotations

from src.clients.curation_opensearch import ClassRegistryError
from src.routers.curation._class_models import (
    SeedConflict,
    SeededClass,
    SeedFromDetectorRequest,
    SeedFromDetectorResponse,
    SeedSkipped,
)
from src.routers.curation._common import OpenSearchDep, get_class_registry, logger, router
from src.routers.curation._config_common_models import api_error
from src.routers.curation._error_models import (
    DetectorUnavailableResponse,
    UnknownDetectorNamesResponse,
)
from src.routers.curation.classes import create_registry_class
from src.routers.curation.ingest import _get_detection_profile
from src.services.curation.detector_vocabulary import detector_labels, plan_seed
from src.services.curation.ingest_detector import effective_profile
from src.services.curation.ingest_policy_store import get_ingest_policy
from src.utils.class_names import class_name_deprecation_index


@router.post(
    '/classes/seed_from_detector',
    response_model=SeedFromDetectorResponse,
    responses={
        422: {'model': UnknownDetectorNamesResponse},
        503: {'model': DetectorUnavailableResponse},
    },
)
async def seed_from_detector(
    payload: SeedFromDetectorRequest, opensearch: OpenSearchDep
) -> SeedFromDetectorResponse:
    """Create registry classes from the ingest detector's labels, by name.

    Append-only and idempotent: a label whose slug (``traffic light`` ->
    ``traffic_light``) already names a class, active or deprecated, is
    skipped, and ids are assigned after the current maximum, never aligned to
    the detector's own class ids. Items are never read or written.
    ``dry_run`` (the default) only reports the plan. ``422`` for a requested
    name the detector does not have; ``503`` when no detector (or no label
    list) is configured.
    """
    try:
        profile = effective_profile(
            _get_detection_profile(), (await get_ingest_policy(opensearch)).detector
        )
        labels = detector_labels(profile) if profile.detector_model else []
    except (ValueError, OSError) as exc:
        raise api_error(503, 'detector_unavailable', f'ingest misconfigured: {exc}') from exc
    if not labels:
        raise api_error(
            503,
            'detector_unavailable',
            'no ingest detector labels available; set OP_INGEST_PRIMARY_DETECTOR_MODEL',
        )
    reg = get_class_registry()
    existing = class_name_deprecation_index(reg.load().classes)
    plan = plan_seed(labels, existing, payload.names)
    if plan.unknown:
        raise api_error(
            422,
            'unknown_detector_names',
            f'names not in the detector vocabulary: {plan.unknown}',
            unknown_names=list(plan.unknown),
        )
    created: list[SeededClass] = []
    skipped = [
        SeedSkipped(name=s.label.slug, detector_label=s.label.name, reason=s.reason)
        for s in plan.skipped
    ]
    for label in plan.create:
        class_id: int | None = None
        if not payload.dry_run:
            try:
                class_id = create_registry_class(
                    reg,
                    name=label.slug,
                    group=payload.group,
                    notes=f'detector label: {label.name}',
                )['class_id']
            except ClassRegistryError:
                # Lost a race with another writer of the same name.
                skipped.append(
                    SeedSkipped(name=label.slug, detector_label=label.name, reason='exists')
                )
                continue
        created.append(SeededClass(class_id=class_id, name=label.slug, detector_label=label.name))
    logger.info(
        'curation_seed_from_detector',
        dry_run=payload.dry_run,
        created=len(created),
        skipped=len(skipped),
    )
    return SeedFromDetectorResponse(
        dry_run=payload.dry_run,
        detector_model=profile.detector_model,
        created=created,
        skipped=skipped,
        conflicts=[
            SeedConflict(
                detector_label=c.label.name,
                class_id_in_detector=c.label.class_id,
                reason=c.reason,
            )
            for c in plan.conflicts
        ],
    )
