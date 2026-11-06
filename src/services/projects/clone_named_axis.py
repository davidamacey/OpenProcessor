"""The stored-config clone axes with one activation (``open_vocab``,
``ingest_profile``): copy a project's stored configs and the activation of the
one in use into another project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.config.project_context import bind_project
from src.services.config_store import get_config_store
from src.services.config_store.documents import (
    ActiveConflictError,
    CloneSource,
    RevisionConflictError,
    activate,
    read_activation,
    save_config,
)
from src.services.projects.clone_stored import copy_stored_configs
from src.services.rejections import rejection


if TYPE_CHECKING:
    from src.config.projects import ProjectRecord
    from src.services.config_store.documents import ConfigAxis, ConfigKind


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
    *, axis: NamedAxis, target_record: ProjectRecord
) -> dict[str, Any] | None:
    """Refuse a target that already has stored sets or an active one (a clone
    is a starting point, never a merge). Returns the ``expected_active`` the
    write must present: the target's REAL activation doc, ``None`` when it
    has none."""
    with bind_project(target_record):
        store = get_config_store()
        await store.ensure_fresh()
        if getattr(store.current, axis.stored_field):
            raise rejection(
                409,
                'target_not_empty',
                f"'{target_record.slug}' already has stored {axis.what}s; "
                f'{axis.axis} cannot be cloned',
                project=target_record.slug,
            )
    existing = await read_activation(target_record.slug, axis.axis)
    if existing is not None and existing.name:
        raise rejection(
            409,
            'target_not_empty',
            f"'{target_record.slug}' already has an active {axis.what}; "
            f'{axis.axis} cannot be cloned',
            project=target_record.slug,
        )
    return existing.as_expected() if existing is not None else None


async def apply_named_axis_clone(
    *,
    axis: NamedAxis,
    source: ProjectRecord,
    target_record: ProjectRecord,
    expected_active: dict[str, Any] | None,
) -> None:
    written = await copy_stored_configs(
        target_record=target_record,
        source=source,
        kind=axis.kind,
        stored_field=axis.stored_field,
    )
    # The body that was ACTIVATED (the immutable revision), not the source's current one: they
    # diverge once the source saves again without reactivating.
    activation = await read_activation(source.slug, axis.axis)
    if activation is None or not activation.name or activation.body is None:
        return
    name, revision, body = activation.name, activation.revision, activation.body
    if revision is None:
        return
    sibling = written.get(name)
    try:
        with bind_project(target_record):
            if sibling is not None and sibling.body == body:
                target_revision = sibling.revision
            else:
                saved, _ = await save_config(
                    kind=axis.kind,
                    name=name,
                    body=body,
                    expected_revision=sibling.revision if sibling else None,
                    cloned_from=CloneSource(source.slug, axis.kind, name, revision),
                )
                target_revision = saved.revision
            await activate(
                axis=axis.axis,
                name=name,
                revision=target_revision,
                expected_active=expected_active,
                reason='clone',
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
