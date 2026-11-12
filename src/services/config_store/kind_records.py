"""The record shape and builders the stored-or-template config kinds share (ingest profiles,
open-vocabulary sets): a document is either stored in the project's config index or a
read-only shipped template, and may be the active one."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from src.services.config_store import documents, get_config_store, templates
from src.services.config_store.documents import TEMPLATE_DESCRIPTION
from src.services.config_store.profiles import RevisionSummary
from src.services.config_store.stored_ops import delete_and_apply, save_and_apply


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from src.services.config_store.documents import Activated, CloneSource


@dataclass(frozen=True)
class KindRecord:
    name: str
    source: Literal['stored', 'template']
    read_only: bool
    revision: int | None
    etag: str
    description: str
    body: dict[str, Any]
    created_at: str | None = None
    updated_at: str | None = None
    cloned_from: str | None = None
    active: bool = False
    active_revision: int | None = None


def stored_record[R: KindRecord](
    cls: type[R],
    *,
    etag_prefix: str,
    stored: Any,
    name: str,
    active_ref: tuple[str | None, int | None],
    read_only: bool,
) -> R:
    """The record of a stored revision ``stored`` (a snapshot or ``documents`` revision)."""
    active_name, active_revision = active_ref
    return cls(
        name=name,
        source='stored',
        read_only=read_only,
        revision=stored.revision,
        etag=f'{etag_prefix}:{name}:{stored.revision}',
        description=stored.description or '',
        body=stored.body,
        created_at=stored.created_at,
        updated_at=stored.updated_at,
        cloned_from=stored.cloned_from,
        active=(active_name == name and active_revision == stored.revision),
        active_revision=active_revision if active_name == name else None,
    )


def build_kind_record[R: KindRecord](
    cls: type[R],
    *,
    etag_prefix: str,
    name: str,
    revision: int | None,
    stored: Any,
    template: dict[str, Any] | None,
    active_ref: tuple[str | None, int | None],
) -> R | None:
    """The current stored record (or the pinned ``revision`` of it), else the shipped template."""
    if stored is not None and (revision is None or revision == stored.revision):
        return stored_record(
            cls,
            etag_prefix=etag_prefix,
            stored=stored,
            name=name,
            active_ref=active_ref,
            read_only=False,
        )
    if template is not None and revision is None:
        body = {k: v for k, v in template.items() if not k.startswith('_')}
        body.pop('name', None)
        digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:12]
        return cls(
            name=name,
            source='template',
            read_only=True,
            revision=None,
            etag=f'{etag_prefix}:{name}:{digest}',
            description=TEMPLATE_DESCRIPTION,
            body=body,
        )
    return None


def revision_summaries(infos: Any) -> list[RevisionSummary]:
    return [
        RevisionSummary(
            revision=i.revision,
            saved_at=i.created_at.isoformat(),
            cloned_from=i.cloned_from,
            description=i.description,
        )
        for i in infos or []
    ]


async def list_kind_revisions(kind: Any, name: str) -> list[RevisionSummary]:
    return revision_summaries(await documents.list_revisions(kind=kind, name=name))


async def save_kind[R: KindRecord](
    kind: Any,
    build: Callable[[str], R | None],
    *,
    name: str,
    body: dict[str, Any],
    expected_revision: int | None,
    description: str,
    cloned_from: CloneSource | None,
) -> R:
    """Save under OCC, apply the new snapshot, and return the record ``build`` serves."""
    await save_and_apply(
        kind,
        name=name,
        body=body,
        expected_revision=expected_revision,
        description=description,
        cloned_from=cloned_from,
    )
    record = build(name)
    assert record is not None
    return record


async def delete_kind(kind: Any, *, name: str, expected_revision: int) -> None:
    await delete_and_apply(kind, name=name, expected_revision=expected_revision)


async def activate_kind(
    axis: Any, *, name: str | None, revision: int | None, expected_active: dict[str, Any] | None
) -> Activated:
    from src.services.config_store.store import activate_axis

    return await activate_axis(
        get_config_store(),
        axis=axis,
        name=name,
        revision=revision,
        expected_active=expected_active,
    )


async def rollback_kind(axis: Any, *, expected_active: dict[str, Any] | None) -> Activated:
    from src.services.config_store.activation_apply import rollback_axis

    return await rollback_axis(get_config_store(), axis=axis, expected_active=expected_active)


async def get_kind_revision_record[R: KindRecord](
    cls: type[R],
    *,
    kind: Any,
    etag_prefix: str,
    name: str,
    revision: int,
    active_ref: tuple[str | None, int | None],
) -> R | None:
    """The read-only record of one stored ``revision``, or ``None`` when it does not exist."""
    stored = await documents.get_revision(kind=kind, name=name, revision=revision)
    if stored is None:
        return None
    return stored_record(
        cls,
        etag_prefix=etag_prefix,
        stored=stored,
        name=name,
        active_ref=active_ref,
        read_only=True,
    )


class StoredKind[R: KindRecord]:
    """The CRUD of one stored-or-template kind: where its stored documents, active ref and
    shipped templates live, and the record class it serves."""

    def __init__(
        self,
        record: type[R],
        *,
        kind: str,
        axis: str,
        etag_prefix: str,
        templates_dir: Path,
        stored: Callable[[Any], dict[str, Any]],
        active: Callable[[Any], Any],
    ) -> None:
        self.record = record
        self.kind = kind
        self.axis = axis
        self.etag_prefix = etag_prefix
        self.templates_dir = templates_dir
        self._stored = stored
        self._active = active

    def load_template(self, name: str) -> dict[str, Any] | None:
        return templates.load_template(self.templates_dir, name)

    def template_names(self) -> list[str]:
        return list(templates.template_names(self.templates_dir))

    def all_known_names(self) -> frozenset[str]:
        return frozenset({*self._stored(get_config_store().current), *self.template_names()})

    def _active_ref(self) -> tuple[str | None, int | None]:
        return templates.active_ref(self._active(get_config_store().current))

    def build_record(self, name: str, *, revision: int | None = None) -> R | None:
        return build_kind_record(
            self.record,
            etag_prefix=self.etag_prefix,
            name=name,
            revision=revision,
            stored=self._stored(get_config_store().current).get(name),
            template=self.load_template(name),
            active_ref=self._active_ref(),
        )

    async def get_revision_record(self, name: str, revision: int) -> R | None:
        return await get_kind_revision_record(
            self.record,
            kind=self.kind,
            etag_prefix=self.etag_prefix,
            name=name,
            revision=revision,
            active_ref=self._active_ref(),
        )

    async def list_revisions(self, name: str) -> list[RevisionSummary] | None:
        if name not in self._stored(get_config_store().current):
            return None
        return await list_kind_revisions(self.kind, name)

    async def save(
        self,
        *,
        name: str,
        body: dict[str, Any],
        expected_revision: int | None,
        description: str = '',
        cloned_from: CloneSource | None = None,
    ) -> R:
        return await save_kind(
            self.kind,
            self.build_record,
            name=name,
            body=body,
            expected_revision=expected_revision,
            description=description,
            cloned_from=cloned_from,
        )

    async def delete(self, *, name: str, expected_revision: int) -> None:
        await delete_kind(self.kind, name=name, expected_revision=expected_revision)

    async def activate(
        self, *, name: str | None, revision: int | None, expected_active: dict[str, Any] | None
    ) -> Activated:
        return await activate_kind(
            self.axis, name=name, revision=revision, expected_active=expected_active
        )

    async def rollback(self, *, expected_active: dict[str, Any] | None) -> Activated:
        return await rollback_kind(self.axis, expected_active=expected_active)
