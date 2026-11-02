"""The stored-config clone axes with one activation (``open_vocab``,
``ingest_profile``): copy a project's stored configs and the activation of the
one in use into another project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.config import get_curation_config
from src.config.project_context import bind_project
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
from src.services.rejections import rejection


if TYPE_CHECKING:
    from src.config.projects import ProjectRecord
    from src.services.config_store.index import ConfigAxis, ConfigKind


@dataclass(frozen=True)
class NamedAxis:
    """One clone axis: its activation ``axis``, stored ``kind``, the snapshot
    field holding the stored configs and the noun used in refusals."""

    axis: ConfigAxis
    kind: ConfigKind
    stored_field: str
    what: str


OPEN_VOCAB = NamedAxis('open_vocab', 'open_vocab_set', 'open_vocab_sets', 'open-vocabulary set')
INGEST_PROFILE = NamedAxis('ingest_profile', 'ingest_profile', 'ingest_profiles', 'ingest profile')


async def validate_named_axis_clone(
    client: Any, *, axis: NamedAxis, target_record: ProjectRecord
) -> dict[str, Any] | None:
    """Refuse a target that already has stored sets or an active one (a clone
    is a starting point, never a merge). Returns the ``expected_active`` the
    write must present: the target's REAL activation doc, ``None`` when it
    has none."""
    with bind_project(target_record):
        store = get_config_store()
        await store.ensure_fresh(client)
        if getattr(store.current, axis.stored_field):
            raise rejection(
                409,
                'target_not_empty',
                f"'{target_record.slug}' already has stored {axis.what}s; "
                f'{axis.axis} cannot be cloned',
                project=target_record.slug,
            )
        existing = await get_activation(client, get_curation_config().configs_index, axis.axis)
    if existing and existing.get('name'):
        raise rejection(
            409,
            'target_not_empty',
            f"'{target_record.slug}' already has an active {axis.what}; "
            f'{axis.axis} cannot be cloned',
            project=target_record.slug,
        )
    return (
        {'name': existing.get('name'), 'revision': existing.get('revision')}
        if existing is not None
        else None
    )


async def apply_named_axis_clone(
    client: Any,
    *,
    axis: NamedAxis,
    source: ProjectRecord,
    target_record: ProjectRecord,
    expected_active: dict[str, Any] | None,
) -> None:
    written = await copy_stored_configs(
        client,
        target_record=target_record,
        source=source,
        kind=axis.kind,
        stored_field=axis.stored_field,
    )
    with bind_project(source, read_only=True):
        source_index = get_curation_config().configs_index
        activation = await get_activation(client, source_index, axis.axis)
        if not activation or not activation.get('name') or activation.get('revision') is None:
            return
        name, revision = activation['name'], int(activation['revision'])
        # The body that was ACTIVATED (the immutable revision copy), not the
        # source's current one: they diverge once the source saves again
        # without reactivating.
        from opensearchpy.exceptions import NotFoundError

        try:
            stored = await client.get(
                index=source_index, id=config_doc_id(axis.kind, name, revision)
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
                    kind=axis.kind,
                    name=name,
                    body=body,
                    expected_revision=sibling.revision if sibling else None,
                    cloned_from=source.slug,
                )
                target_revision = doc['revision']
            await activate(
                client,
                target_index,
                axis=axis.axis,
                name=name,
                revision=target_revision,
                expected_active=expected_active,
            )
    except (RevisionConflictError, ActiveConflictError) as exc:
        raise rejection(
            409,
            'target_not_empty',
            f"'{target_record.slug}' changed while {axis.axis} was being cloned",
            project=target_record.slug,
        ) from exc


__all__ = [
    'INGEST_PROFILE',
    'OPEN_VOCAB',
    'NamedAxis',
    'apply_named_axis_clone',
    'validate_named_axis_clone',
]
