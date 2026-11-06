"""The ONE module that calls the stage-2 repository load functions (D-84, the upgrade load
contract of lanes 2A-2D).

Every control row of the upgrade is written through these functions inside the caller's
``deployment_tx``: they take the caller's session, preserve the given ids and timestamps, write
exactly the rows the normal write path writes (same normalization, same derived rows), call no
OpenSearch and commit nothing. When a repository signature changes this is the only file to edit.

    2A  src/db/repos/projects.py   ProjectLoad, load_project
    2B  src/db/repos/classes.py    ClassLoad, load_registry
    2C  src/db/repos/config.py     ConfigRevisionLoad, ConfigDocumentLoad, load_config_document,
                                   ActivationEventLoad, load_activation_events
        src/db/repos/settings.py   load_setting, load_keymap
    2D  src/db/repos/records.py    HoldoutFreezeLoad, load_holdout_freezes, LabelOperationLoad,
                                   LabelOperationMemberLoad, load_label_operation, load_project_state
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import uuid
    from collections.abc import Iterable, Mapping, Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.services.upgrade.from_060.model import (
        ActivationSrc,
        ConfigDocSrc,
        HoldoutSrc,
        KeymapSrc,
        LabelOpSrc,
        ProjectSrc,
        RegistrySrc,
        SettingsSrc,
    )

CREATED_BY = 'upgrade_from_060'


def _repo(name: str) -> Any:
    # Imported on use: the repositories land with lanes 2A-2D, and the module that calls them is
    # this one only.
    return importlib.import_module(f'src.db.repos.{name}')


async def load_project(session: AsyncSession, p: ProjectSrc) -> uuid.UUID:
    projects = _repo('projects')
    row = projects.ProjectLoad(
        id=p.id,
        slug=p.slug,
        display_name=p.display_name,
        description=p.description,
        status=p.status,
        pre_delete_status=p.pre_delete_status,
        revision=p.revision,
        paused_at=p.paused_at,
        origin=p.origin,
        index_prefix=p.index_prefix,
        storage_prefix=p.storage_prefix,
        model_prefix=p.model_prefix,
        mlflow_experiment=p.mlflow_experiment,
        created_at=p.created_at,
        updated_at=p.updated_at,
        archived_at=p.updated_at if p.status == 'archived' else None,
        created_by=CREATED_BY,
        indexes=dict(p.indexes),
    )
    result: uuid.UUID = await projects.load_project(session, row)
    return result


async def load_registry(session: AsyncSession, project_id: uuid.UUID, reg: RegistrySrc) -> None:
    classes = _repo('classes')
    rows = [
        classes.ClassLoad(
            class_id=c.class_id,
            name=c.name,
            class_group=c.class_group,
            deprecated=c.deprecated,
            merged_into=c.merged_into,
            hotkey_letter=c.hotkey_letter,
            notes=c.notes,
            added_at=c.added_at,
            updated_at=c.added_at,
        )
        for c in reg.classes
    ]
    await classes.load_registry(
        session,
        project_id,
        rows,
        revision=reg.revision,
        next_class_id=reg.next_class_id,
        migrated_from=reg.migrated_from,
        migrated_at=reg.migrated_at,
    )


async def load_config(
    session: AsyncSession, project_id: uuid.UUID | None, doc: ConfigDocSrc
) -> uuid.UUID:
    config = _repo('config')
    revisions = [
        config.ConfigRevisionLoad(
            revision=r.revision, body=r.body, created_at=r.created_at, cloned_from=r.cloned_from
        )
        for r in doc.revisions
    ]
    row = config.ConfigDocumentLoad(
        project_id=project_id, kind=doc.kind, name=doc.name, revisions=revisions
    )
    result: uuid.UUID = await config.load_config_document(session, row)
    return result


async def load_activations(
    session: AsyncSession,
    project_id: uuid.UUID,
    events: Sequence[ActivationSrc],
    config_ids: Mapping[tuple[str, str], uuid.UUID],
) -> None:
    """The current activation of each axis is its last event."""
    config = _repo('config')
    rows = [
        config.ActivationEventLoad(
            axis=e.axis,
            from_config_id=config_ids.get((e.kind, e.from_name)) if e.from_name else None,
            from_revision=e.from_revision if e.from_name else None,
            to_config_id=config_ids[(e.kind, e.to_name)],
            to_revision=e.to_revision,
            reason='upgrade',
            at=e.at,
            by=CREATED_BY,
        )
        for e in events
        if (e.kind, e.to_name) in config_ids
    ]
    await config.load_activation_events(session, project_id, rows)


async def load_settings(
    session: AsyncSession, project_id: uuid.UUID, settings: SettingsSrc
) -> None:
    repo = _repo('settings')
    await repo.load_setting(session, project_id, 'defaults', settings.defaults, revision=1)
    for key, body in settings.policies.items():
        await repo.load_setting(
            session, project_id, key, body, revision=max(1, int(body.get('revision') or 1))
        )


async def load_keymap(session: AsyncSession, project_id: uuid.UUID, keymap: KeymapSrc) -> None:
    await _repo('settings').load_keymap(
        session, project_id, keymap.overrides, revision=keymap.revision
    )


async def load_holdouts(
    session: AsyncSession, project_id: uuid.UUID, holdouts: Iterable[HoldoutSrc]
) -> None:
    records = _repo('records')
    rows = [
        records.HoldoutFreezeLoad(
            id=h.id,
            is_current=h.is_current,
            source=h.source,
            item_count=h.item_count,
            seed=None,
            snapshot_storage_key=h.snapshot_key,
            frozen_at=h.frozen_at,
            frozen_by=h.frozen_by,
        )
        for h in holdouts
    ]
    await records.load_holdout_freezes(session, project_id, rows)


async def load_label_op(
    session: AsyncSession, project_id: uuid.UUID, op: LabelOpSrc, member_ids: Iterable[str]
) -> None:
    """One operation and its members (the repository writes them in batches of 5,000, trap T-14)."""
    records = _repo('records')
    row = records.LabelOperationLoad(
        id=op.id,
        project_id=project_id,
        kind=op.kind,
        status=op.status,
        params=op.params,
        member_count=0,
        job_id=None,
        created_at=op.created_at,
        created_by=op.created_by,
        undone_at=op.undone_at,
    )
    members = (records.LabelOperationMemberLoad(item_id=i, prior={}) for i in member_ids)
    await records.load_label_operation(session, row, members)


async def load_state_marker(
    session: AsyncSession, project_id: uuid.UUID, key: str, state: dict[str, Any]
) -> None:
    """A ``project_state`` row: ``cluster_refresh``, ``region_drain`` or ``region_dependency_health``."""
    body = {k: v for k, v in state.items() if k != 'revision'}
    await _repo('records').load_project_state(
        session, project_id, key, body, revision=max(1, int(state.get('revision') or 1))
    )


async def ensure_class(
    session: AsyncSession, project_id: uuid.UUID, name: str, *, group: str, notes: str
) -> int:
    """The id of the active class ``name``, added first when the registry lacks it (2B's
    ``ensure_class``: the one allocation path)."""
    result: int = await _repo('classes').ensure_class(
        session, project_id, name, group=group, notes=notes
    )
    return result
