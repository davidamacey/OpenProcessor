"""The ``open_vocab`` clone axis: copy a project's stored prompt sets and the
activation of the one in use into another project."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.config import get_curation_config
from src.config.project_context import bind_project
from src.routers.curation._config_common_models import api_error
from src.services.config_store import get_config_store
from src.services.config_store.index import (
    ActiveConflictError,
    RevisionConflictError,
    activate,
    config_doc_id,
    get_activation,
    save_config,
)
from src.services.projects.clone_stored import copy_stored_configs


if TYPE_CHECKING:
    from src.config.projects import ProjectRecord


async def validate_open_vocab_clone(
    client: Any, *, target_record: ProjectRecord
) -> dict[str, Any] | None:
    """Refuse a target that already has stored sets or an active one (a clone
    is a starting point, never a merge). Returns the ``expected_active`` the
    write must present: the target's REAL activation doc, ``None`` when it
    has none."""
    with bind_project(target_record):
        store = get_config_store()
        await store.ensure_fresh(client)
        if store.current.open_vocab_sets:
            raise api_error(
                409,
                'target_not_empty',
                f"'{target_record.slug}' already has stored open-vocabulary sets; "
                'open_vocab cannot be cloned',
                project=target_record.slug,
            )
        existing = await get_activation(client, get_curation_config().configs_index, 'open_vocab')
    if existing and existing.get('name'):
        raise api_error(
            409,
            'target_not_empty',
            f"'{target_record.slug}' already has an active open-vocabulary set; "
            'open_vocab cannot be cloned',
            project=target_record.slug,
        )
    return (
        {'name': existing.get('name'), 'revision': existing.get('revision')}
        if existing is not None
        else None
    )


async def apply_open_vocab_clone(
    client: Any,
    *,
    source: ProjectRecord,
    target_record: ProjectRecord,
    expected_active: dict[str, Any] | None,
) -> None:
    written = await copy_stored_configs(
        client,
        target_record=target_record,
        source=source,
        kind='open_vocab_set',
        stored_field='open_vocab_sets',
    )
    with bind_project(source, read_only=True):
        source_index = get_curation_config().configs_index
        activation = await get_activation(client, source_index, 'open_vocab')
        if not activation or not activation.get('name') or activation.get('revision') is None:
            return
        name, revision = activation['name'], int(activation['revision'])
        # The body that was ACTIVATED (the immutable revision copy), not the
        # source's current one: they diverge once the source saves again
        # without reactivating.
        from opensearchpy.exceptions import NotFoundError

        try:
            stored = await client.get(
                index=source_index, id=config_doc_id('open_vocab_set', name, revision)
            )
        except NotFoundError:
            return
    body = (stored.get('_source') or {}).get('body')
    if body is None:
        return
    sibling = written.get(name)
    try:
        with bind_project(target_record):
            target_index = get_curation_config().configs_index
            if sibling is not None and sibling.body == body:
                target_revision = sibling.revision
            else:
                doc = await save_config(
                    client,
                    target_index,
                    kind='open_vocab_set',
                    name=name,
                    body=body,
                    expected_revision=sibling.revision if sibling else None,
                    cloned_from=source.slug,
                )
                target_revision = doc['revision']
            await activate(
                client,
                target_index,
                axis='open_vocab',
                name=name,
                revision=target_revision,
                expected_active=expected_active,
            )
    except (RevisionConflictError, ActiveConflictError) as exc:
        raise api_error(
            409,
            'target_not_empty',
            f"'{target_record.slug}' changed while open_vocab was being cloned",
            project=target_record.slug,
        ) from exc


__all__ = ['apply_open_vocab_clone', 'validate_open_vocab_clone']
