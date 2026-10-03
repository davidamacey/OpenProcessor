"""``POST /region_profiles/{name}/clone`` -- split out of
``region_profiles.py`` to stay under the repo's 700-LOC pre-commit
ratchet (same pattern as ``_models_sharing.py``/``models.py``).
"""

from __future__ import annotations

from typing import Any

from src.routers.curation._common import OpenSearchDep, router
from src.routers.curation._config_common_models import api_error
from src.routers.curation._region_profile_models import RegionProfileCloneRequest, RegionProfileDoc
from src.services.config_store import RevisionConflictError
from src.services.config_store.clone_shared import (
    cloned_description,
    cloned_from_tag,
    read_source_record,
    reject_invalid_clone_name,
)
from src.services.config_store.profile_validation import validate_profile
from src.services.config_store.profiles import (
    ProfileRecord,
    all_known_names,
    build_record,
    get_revision_record,
    save_profile,
)


async def _resolve_clone_source(
    client: Any, *, name: str, revision: int | None, source: str | None
) -> ProfileRecord:
    if source == 'stored' and revision is not None:
        record = await get_revision_record(client, name, revision)
        if record is None:
            raise api_error(404, 'unknown_revision', f'{name!r} has no revision {revision}')
        return record
    if source is not None:
        record = build_record(name, revision=revision)
        if record is None or record.source != source:
            raise api_error(404, 'not_found', f'{name!r} has no {source} source')
        return record
    if revision is not None:
        record = await get_revision_record(client, name, revision)
        if record is not None:
            return record
    record = build_record(name, revision=revision)
    if record is None:
        raise api_error(404, 'not_found', f'{name!r} is not a known region profile')
    return record


@router.post('/region_profiles/{name}/clone', response_model=RegionProfileDoc, status_code=201)
async def clone_region_profile(
    name: str, body: RegionProfileCloneRequest, opensearch: OpenSearchDep
) -> RegionProfileDoc:
    from src.config import get_curation_config
    from src.routers.curation._models_segmenter import configured_segmenter_health
    from src.routers.curation.region_profiles import _registry_class_names, _to_doc

    target_slug = get_curation_config().project_slug

    async def _resolve(client: Any) -> ProfileRecord:
        return await _resolve_clone_source(
            client, name=name, revision=body.revision, source=body.source
        )

    source = await read_source_record(
        from_project=body.from_project,
        target_slug=target_slug,
        opensearch=opensearch,
        resolve=_resolve,
    )

    existing = all_known_names()
    # M-2 fix: new_name must obey the same slug/reserved-word rules as create.
    name_report = await validate_profile(
        body.new_name,
        source.body,
        existing_names=existing,
        segmenter_health=configured_segmenter_health,
        class_names=_registry_class_names(),
        project_slug=target_slug,
    )
    reject_invalid_clone_name(
        name_report,
        new_name=body.new_name,
        existing=existing,
        name_codes=('profile_name_invalid', 'profile_name_reserved'),
        what='profile',
    )

    new_body = dict(source.body)
    report = await validate_profile(
        None,
        new_body,
        segmenter_health=configured_segmenter_health,
        class_names=_registry_class_names(),
        project_slug=target_slug,
    )
    if not report.ok:
        raise api_error(
            422, 'validation_failed', 'the cloned profile has content errors', report=report
        )

    cloned_from = cloned_from_tag(
        source_project=body.from_project or target_slug, name=name, revision=source.revision
    )
    try:
        record = await save_profile(
            opensearch,
            name=body.new_name,
            body=new_body,
            expected_revision=None,
            description=cloned_description(body.description, source.description),
            cloned_from=cloned_from,
        )
    except RevisionConflictError as exc:
        # M-5 fix: a name visible in OpenSearch but not yet to this
        # process's stale/cold store snapshot must still 409, not 500.
        raise api_error(409, 'name_conflict', f'{body.new_name!r} is already taken') from exc
    return _to_doc(record, validation=report)


__all__ = ['clone_region_profile']
