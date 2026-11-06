"""``POST /ingest_profiles/{name}/clone``: copy a profile (or a shipped template,
or a profile of another project) under a new name."""

from __future__ import annotations

from src.routers.curation._api_error import api_error
from src.routers.curation._common import router
from src.routers.curation._ingest_profile_models import IngestProfileCloneRequest, IngestProfileDoc
from src.services.config_store import RevisionConflictError
from src.services.config_store.clone_shared import (
    cloned_description,
    read_source_record,
    reject_invalid_clone_name,
)
from src.services.config_store.documents import CloneSource
from src.services.config_store.ingest_profile_validation import validate_ingest_profile
from src.services.config_store.ingest_profiles import (
    IngestProfileRecord,
    all_known_names,
    build_record,
    get_revision_record,
    save_profile,
)


async def _resolve_clone_source(
    *, name: str, revision: int | None, source: str | None
) -> IngestProfileRecord:
    if revision is not None and source in (None, 'stored'):
        record = await get_revision_record(name, revision)
        if record is not None:
            return record
        if source == 'stored':
            raise api_error(404, 'unknown_revision', f'{name!r} has no revision {revision}')
    record = build_record(name, revision=revision)
    if record is None or (source is not None and record.source != source):
        raise api_error(404, 'not_found', f'{name!r} is not a known ingest profile')
    return record


@router.post('/ingest_profiles/{name}/clone', response_model=IngestProfileDoc, status_code=201)
async def clone_ingest_profile(name: str, body: IngestProfileCloneRequest) -> IngestProfileDoc:
    from src.config import get_curation_config
    from src.routers.curation.ingest_profiles import _to_doc

    target_slug = get_curation_config().project_slug

    async def _resolve() -> IngestProfileRecord:
        return await _resolve_clone_source(name=name, revision=body.revision, source=body.source)

    source = await read_source_record(
        from_project=body.from_project,
        target_slug=target_slug,
        resolve=_resolve,
    )
    existing = all_known_names()
    reject_invalid_clone_name(
        await validate_ingest_profile(body.new_name, source.body, existing_names=existing),
        new_name=body.new_name,
        existing=existing,
        name_codes=('ingest_profile_name_invalid', 'ingest_profile_name_reserved'),
        what='profile',
    )
    report = await validate_ingest_profile(None, source.body)
    if not report.ok:
        raise api_error(
            422, 'validation_failed', 'the cloned profile has content errors', report=report
        )
    try:
        record = await save_profile(
            name=body.new_name,
            body=dict(source.body),
            expected_revision=None,
            description=cloned_description(body.description, source.description),
            cloned_from=CloneSource(
                body.from_project or target_slug, 'ingest_profile', name, source.revision
            ),
        )
    except RevisionConflictError as exc:
        raise api_error(409, 'name_conflict', f'{body.new_name!r} is already taken') from exc
    return _to_doc(record, validation=report)


__all__ = ['clone_ingest_profile']
