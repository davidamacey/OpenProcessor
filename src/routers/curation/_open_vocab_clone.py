"""``POST /open_vocab/{name}/clone``: copy a set (or a shipped template, or a
set of another project) under a new name. Split out of ``open_vocab.py`` for
the 700-LOC ratchet, like ``_region_profile_clone.py``."""

from __future__ import annotations

from typing import Any

from src.routers.curation._common import OpenSearchDep, router
from src.routers.curation._config_common_models import api_error
from src.routers.curation._open_vocab_models import OpenVocabCloneRequest, OpenVocabDoc
from src.services.config_store import RevisionConflictError
from src.services.config_store.clone_shared import (
    cloned_from_tag,
    read_source_record,
    reject_invalid_clone_name,
)
from src.services.config_store.open_vocab import (
    OpenVocabRecord,
    all_known_names,
    build_record,
    get_revision_record,
    save_set,
)
from src.services.config_store.open_vocab_validation import validate_open_vocab


async def _resolve_clone_source(
    client: Any, *, name: str, revision: int | None, source: str | None
) -> OpenVocabRecord:
    if revision is not None and source in (None, 'stored'):
        record = await get_revision_record(client, name, revision)
        if record is not None:
            return record
        if source == 'stored':
            raise api_error(404, 'unknown_revision', f'{name!r} has no revision {revision}')
    record = build_record(name, revision=revision)
    if record is None or (source is not None and record.source != source):
        raise api_error(404, 'not_found', f'{name!r} is not a known open-vocabulary set')
    return record


@router.post('/open_vocab/{name}/clone', response_model=OpenVocabDoc, status_code=201)
async def clone_open_vocab(
    name: str, body: OpenVocabCloneRequest, opensearch: OpenSearchDep
) -> OpenVocabDoc:
    from src.config import get_curation_config
    from src.routers.curation.open_vocab import _to_doc, validation_inputs

    target_slug = get_curation_config().project_slug

    async def _resolve(client: Any) -> OpenVocabRecord:
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
    inputs = validation_inputs()
    reject_invalid_clone_name(
        await validate_open_vocab(body.new_name, source.body, existing_names=existing, **inputs),
        new_name=body.new_name,
        existing=existing,
        name_codes=('open_vocab_name_invalid', 'open_vocab_name_reserved'),
        what='set',
    )
    report = await validate_open_vocab(None, source.body, **inputs)
    if not report.ok:
        raise api_error(
            422, 'validation_failed', 'the cloned set has content errors', report=report
        )
    try:
        record = await save_set(
            opensearch,
            name=body.new_name,
            body=dict(source.body),
            expected_revision=None,
            description=body.description if body.description is not None else source.description,
            cloned_from=cloned_from_tag(
                source_project=body.from_project or target_slug,
                name=name,
                revision=source.revision,
            ),
        )
    except RevisionConflictError as exc:
        raise api_error(409, 'name_conflict', f'{body.new_name!r} is already taken') from exc
    return _to_doc(record, validation=report)


__all__ = ['clone_open_vocab']
