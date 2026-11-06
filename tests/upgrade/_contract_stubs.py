"""Stand-ins for the upgrade load contract of lanes 2A-2D, used only while those lanes are not merged.

``tests/upgrade/conftest.py`` installs these as ``src.db.repos.*`` when the real package is
absent, so the upgrade's own code (parsers, orchestration, COPY loads, steps, verify, delete,
rollback) is tested now. They implement the contract's names and field spellings with plain ORM
writes. When the real repositories are merged ``find_spec`` finds them, this module is not
installed and is deleted.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import sys
import types
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert

from src.db.models import (
    ClassRegistry,
    ConfigActivation,
    ConfigActivationEvent,
    ConfigDocument,
    ConfigRevision,
    DeploymentCounter,
    HoldoutFreeze,
    LabelOperation,
    LabelOperationMember,
    Project,
    ProjectClass,
    ProjectIndex,
    ProjectKeymap,
    ProjectSettings,
    ProjectSlug,
    ProjectState,
    VlmEndpointProbe,
)
from src.utils.class_names import normalize_class_name


if TYPE_CHECKING:
    from datetime import datetime


def _module(name: str, **members: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(members)
    sys.modules[name] = module
    parent, _, leaf = name.rpartition('.')
    if parent in sys.modules:
        setattr(sys.modules[parent], leaf, module)
    return module


def install() -> None:
    @dataclass(frozen=True)
    class ProjectLoad:
        slug: str
        display_name: str
        description: str
        status: str
        revision: int
        created_at: datetime
        updated_at: datetime
        index_prefix: str
        storage_prefix: str
        model_prefix: str
        mlflow_experiment: str
        indexes: dict[str, str]
        created_by: str
        pre_delete_status: str | None = None
        config_epoch: int = 0
        paused_at: datetime | None = None
        paused_by: str | None = None
        origin: Any = None
        created_by_user_id: uuid.UUID | None = None

    async def load_project(session: Any, row: ProjectLoad) -> uuid.UUID:
        session.add(ProjectSlug(slug=row.slug, created_at=row.created_at))
        await session.flush()
        project = Project(
            slug=row.slug, display_name=row.display_name, description=row.description,
            status=row.status, pre_delete_status=row.pre_delete_status, revision=row.revision,
            paused_at=row.paused_at, origin=row.origin, index_prefix=row.index_prefix,
            storage_prefix=row.storage_prefix, model_prefix=row.model_prefix,
            mlflow_experiment=row.mlflow_experiment, created_at=row.created_at,
            updated_at=row.updated_at, created_by=row.created_by,
        )  # fmt: skip
        session.add(project)
        await session.flush()
        project_id = project.id
        for role, name in row.indexes.items():
            session.add(ProjectIndex(project_id=project_id, role=role, index_name=name))
        session.add(ClassRegistry(project_id=project_id))  # the create path's empty registry row
        session.add(ProjectSettings(project_id=project_id, updated_by=row.created_by))
        session.add(ProjectKeymap(project_id=project_id, updated_by=row.created_by))
        stmt = insert(DeploymentCounter).values(name='projects_revision', value=1)
        await session.execute(
            stmt.on_conflict_do_update(
                index_elements=[DeploymentCounter.name], set_={'value': DeploymentCounter.value + 1}
            )
        )
        await session.flush()
        return project_id

    importlib.import_module('src.db.repos')  # the package exists (lane ES-A's repository is in it)
    _module('src.db.repos.projects', ProjectLoad=ProjectLoad, load_project=load_project)

    @dataclass(frozen=True)
    class ClassLoad:
        class_id: int
        name: str
        class_group: str | None
        deprecated: bool
        merged_into: int | None
        hotkey_letter: str | None
        notes: str
        added_at: datetime
        updated_at: datetime

    async def load_registry(
        session: Any,
        project_id: uuid.UUID,
        classes: Any,
        *,
        revision: int,
        next_class_id: int | None = None,
        migrated_from: str | None = None,
        migrated_at: datetime | None = None,
    ) -> None:
        rows = list(classes)
        top = (
            next_class_id
            if next_class_id is not None
            else max((c.class_id for c in rows), default=-1) + 1
        )
        found = await session.execute(
            select(ClassRegistry).where(ClassRegistry.project_id == project_id)
        )
        if found.first() is not None:
            raise RuntimeError(f'project {project_id} already has a class registry')
        session.add(
            ClassRegistry(
                project_id=project_id,
                next_class_id=top,
                revision=revision,
                migrated_from=migrated_from,
                migrated_at=migrated_at,
            )
        )
        await session.flush()
        for c in rows:
            session.add(
                ProjectClass(
                    project_id=project_id, class_id=c.class_id, name=c.name, class_group=c.class_group,
                    deprecated=c.deprecated, hotkey_letter=c.hotkey_letter, notes=c.notes,
                    added_at=c.added_at, updated_at=c.updated_at, name_key=normalize_class_name(c.name),
                )
            )  # fmt: skip
        await session.flush()
        for c in rows:  # merged_into references a sibling, so it is set after every row exists
            if c.merged_into is not None:
                await session.execute(
                    update(ProjectClass)
                    .where(
                        ProjectClass.project_id == project_id, ProjectClass.class_id == c.class_id
                    )
                    .values(merged_into=c.merged_into)
                )

    async def ensure_class(
        session: Any, project_id: uuid.UUID, name: str, *, group: str, notes: str
    ) -> int:
        key = normalize_class_name(name)
        found = (
            await session.execute(
                select(ProjectClass.class_id).where(
                    ProjectClass.project_id == project_id,
                    ProjectClass.name_key == key,
                    ProjectClass.deprecated.is_(False),
                )
            )
        ).scalar_one_or_none()
        if found is not None:
            return int(found)
        registry = (
            await session.execute(
                select(ClassRegistry).where(ClassRegistry.project_id == project_id)
            )
        ).scalar_one()
        new_id = registry.next_class_id
        session.add(
            ProjectClass(
                project_id=project_id,
                class_id=new_id,
                name=name,
                class_group=group,
                notes=notes,
                name_key=key,
            )
        )
        registry.next_class_id = new_id + 1
        await session.flush()
        return int(new_id)

    _module(
        'src.db.repos.classes',
        ClassLoad=ClassLoad,
        load_registry=load_registry,
        ensure_class=ensure_class,
    )

    off_config_id = uuid.UUID(int=0)

    @dataclass(frozen=True)
    class ConfigRevisionLoad:
        revision: int
        body: Any
        created_at: datetime
        created_by: str = 'upgrade'

    @dataclass(frozen=True)
    class ConfigDocumentLoad:
        project_id: uuid.UUID | None
        kind: str
        name: str
        revisions: list[ConfigRevisionLoad] = field(default_factory=list)
        description: str = ''
        id: uuid.UUID | None = None
        cloned_from_id: uuid.UUID | None = None
        created_at: datetime | None = None
        updated_at: datetime | None = None
        updated_by: str = 'upgrade'

    @dataclass(frozen=True)
    class ActivationEventLoad:
        axis: str
        to_config_id: uuid.UUID
        to_revision: int
        at: datetime
        from_config_id: uuid.UUID | None = None
        from_revision: int | None = None
        reason: str = 'upgrade'
        by: str = 'upgrade'

    @dataclass(frozen=True)
    class ProbeLoad:
        revision: int
        probed_at: datetime
        ok: bool
        result: Any

    @dataclass(frozen=True)
    class VlmEndpointLoad:
        document: ConfigDocumentLoad
        probes: list[ProbeLoad] = field(default_factory=list)

    async def load_config_document(session: Any, doc: ConfigDocumentLoad) -> uuid.UUID:
        revisions = sorted(doc.revisions, key=lambda r: r.revision)
        config_id = doc.id or uuid.uuid4()
        session.add(
            ConfigDocument(
                id=config_id, project_id=doc.project_id, kind=doc.kind, name=doc.name,
                current_revision=revisions[-1].revision, description=doc.description,
                created_at=doc.created_at or revisions[0].created_at,
                updated_at=doc.updated_at or revisions[-1].created_at, updated_by=doc.updated_by,
            )
        )  # fmt: skip
        await session.flush()
        for r in revisions:
            digest = hashlib.sha256(json.dumps(r.body, sort_keys=True).encode()).hexdigest()
            session.add(
                ConfigRevision(
                    config_id=config_id,
                    revision=r.revision,
                    body=r.body,
                    body_sha256=digest,
                    created_at=r.created_at,
                    created_by=r.created_by,
                )
            )
        await session.flush()
        return config_id

    async def load_vlm_endpoint(session: Any, row: VlmEndpointLoad) -> None:
        config_id = await load_config_document(session, row.document)
        for p in row.probes:
            session.add(
                VlmEndpointProbe(
                    config_id=config_id,
                    revision=p.revision,
                    probed_at=p.probed_at,
                    ok=p.ok,
                    result=p.result,
                )
            )
        await session.flush()

    async def load_activation_events(
        session: Any, project_id: uuid.UUID | None, events: Any
    ) -> None:
        last: dict[str, Any] = {}
        for e in events:
            session.add(
                ConfigActivationEvent(
                    project_id=project_id, axis=e.axis, from_config_id=e.from_config_id,
                    from_revision=e.from_revision, to_config_id=e.to_config_id, to_revision=e.to_revision,
                    reason=e.reason, at=e.at, by=e.by,
                )
            )  # fmt: skip
            last[e.axis] = e
        await session.flush()
        for axis, e in last.items():
            if e.to_config_id == off_config_id:
                continue
            kind = (
                await session.execute(
                    select(ConfigDocument.kind).where(ConfigDocument.id == e.to_config_id)
                )
            ).scalar_one()
            session.add(
                ConfigActivation(
                    project_id=project_id,
                    axis=axis,
                    kind=kind,
                    config_id=e.to_config_id,
                    revision=e.to_revision,
                    activated_at=e.at,
                    activated_by=e.by,
                )
            )
        await session.flush()

    _module(
        'src.db.repos.config',
        ConfigRevisionLoad=ConfigRevisionLoad, ConfigDocumentLoad=ConfigDocumentLoad,
        ActivationEventLoad=ActivationEventLoad, ProbeLoad=ProbeLoad, VlmEndpointLoad=VlmEndpointLoad,
        OFF_CONFIG_ID=off_config_id, load_config_document=load_config_document,
        load_vlm_endpoint=load_vlm_endpoint, load_activation_events=load_activation_events,
    )  # fmt: skip

    async def load_setting(
        session: Any, project_id: uuid.UUID, key: str, body: Any, *, revision: int
    ) -> None:
        await session.execute(
            update(ProjectSettings)
            .where(ProjectSettings.project_id == project_id)
            .values({key: body, 'revision': revision})
        )

    async def load_keymap(session: Any, project_id: uuid.UUID, body: Any, *, revision: int) -> None:
        await session.execute(
            update(ProjectKeymap)
            .where(ProjectKeymap.project_id == project_id)
            .values(overrides=body, revision=revision)
        )

    _module('src.db.repos.settings', load_setting=load_setting, load_keymap=load_keymap)

    @dataclass(frozen=True)
    class HoldoutFreezeLoad:
        id: uuid.UUID
        is_current: bool
        source: str
        item_count: int
        seed: int | None
        snapshot_storage_key: str | None
        frozen_at: datetime
        frozen_by: str

    @dataclass(frozen=True)
    class LabelOperationLoad:
        id: uuid.UUID
        project_id: uuid.UUID
        kind: str
        status: str
        params: Any
        member_count: int
        job_id: uuid.UUID | None
        created_at: datetime
        created_by: str
        undone_at: datetime | None

    @dataclass(frozen=True)
    class LabelOperationMemberLoad:
        item_id: str
        prior: Any

    async def load_holdout_freezes(session: Any, project_id: uuid.UUID, rows: Any) -> None:
        for r in rows:
            session.add(HoldoutFreeze(**dataclasses.asdict(r), project_id=project_id))
        await session.flush()

    async def load_label_operation(session: Any, op: LabelOperationLoad, members: Any) -> None:
        rows = list(members)
        session.add(LabelOperation(**{**dataclasses.asdict(op), 'member_count': len(rows)}))
        await session.flush()
        for start in range(0, len(rows), 5000):
            session.add_all(
                LabelOperationMember(op_id=op.id, item_id=m.item_id, prior=m.prior)
                for m in rows[start : start + 5000]
            )
            await session.flush()

    async def load_project_state(
        session: Any, project_id: uuid.UUID, key: str, body: Any, *, revision: int
    ) -> None:
        session.add(ProjectState(project_id=project_id, key=key, value=body, revision=revision))
        await session.flush()

    _module(
        'src.db.repos.records',
        HoldoutFreezeLoad=HoldoutFreezeLoad, LabelOperationLoad=LabelOperationLoad,
        LabelOperationMemberLoad=LabelOperationMemberLoad, load_holdout_freezes=load_holdout_freezes,
        load_label_operation=load_label_operation, load_project_state=load_project_state,
    )  # fmt: skip
