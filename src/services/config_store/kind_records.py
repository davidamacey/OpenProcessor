"""The record shape and builders the stored-or-template config kinds share (ingest profiles,
open-vocabulary sets): a document is either stored in the project's config index or a
read-only shipped template, and may be the active one."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

from src.services.config_store.documents import TEMPLATE_DESCRIPTION
from src.services.config_store.profiles import RevisionSummary


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
