"""Ingest detector profile CRUD (the ``ingest_profile`` kind / axis). Mirrors
:mod:`src.services.config_store.open_vocab`: a profile is either stored in the
project's config index or a read-only shipped template
(``examples/ingest_profiles/``). The deployment's ``OP_INGEST_*`` profile is the
default a project with no active profile runs; it is not a stored profile.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from src.services.config_store import documents, get_config_store
from src.services.config_store.documents import (
    TEMPLATE_DESCRIPTION,
    ActiveConflictError,
    CloneSource,
    RevisionConflictError,
)
from src.services.config_store.profiles import RevisionSummary
from src.services.config_store.stored_ops import delete_and_apply, save_and_apply


if TYPE_CHECKING:
    from src.services.config_store.documents import Activated

ProfileSource = Literal['stored', 'template']

_TEMPLATES_DIR = Path(__file__).resolve().parents[3] / 'examples' / 'ingest_profiles'
_KIND: Literal['ingest_profile'] = 'ingest_profile'


@dataclass(frozen=True)
class IngestProfileRecord:
    name: str
    source: ProfileSource
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


def _template_names() -> dict[str, Path]:
    if not _TEMPLATES_DIR.is_dir():
        return {}
    return {p.stem: p for p in sorted(_TEMPLATES_DIR.glob('*.json'))}


def load_template(name: str) -> dict[str, Any] | None:
    path = _template_names().get(name)
    if path is None:
        return None
    try:
        typed: dict[str, Any] | None = json.loads(path.read_text(encoding='utf-8'))
        return typed
    except (OSError, ValueError):
        return None


def template_names() -> list[str]:
    return list(_template_names())


def all_known_names() -> frozenset[str]:
    return frozenset({*get_config_store().current.ingest_profiles, *_template_names()})


def _active_ref() -> tuple[str | None, int | None]:
    ref = get_config_store().current.active_ingest_profile
    if ref is None or ref == 'off':
        return None, None
    return ref


def _stored_record(
    name: str, *, revision: int, body: dict[str, Any], src: dict[str, Any], read_only: bool
) -> IngestProfileRecord:
    active_name, active_revision = _active_ref()
    return IngestProfileRecord(
        name=name,
        source='stored',
        read_only=read_only,
        revision=revision,
        etag=f'ingest_profile:{name}:{revision}',
        description=src.get('description') or '',
        body=body,
        created_at=src.get('created_at'),
        updated_at=src.get('updated_at'),
        cloned_from=src.get('cloned_from'),
        active=(active_name == name and active_revision == revision),
        active_revision=active_revision if active_name == name else None,
    )


def build_record(name: str, *, revision: int | None = None) -> IngestProfileRecord | None:
    stored = get_config_store().current.ingest_profiles.get(name)
    if stored is not None and (revision is None or revision == stored.revision):
        return _stored_record(
            name,
            revision=stored.revision,
            body=stored.body,
            src={
                'description': stored.description,
                'created_at': stored.created_at,
                'updated_at': stored.updated_at,
                'cloned_from': stored.cloned_from,
            },
            read_only=False,
        )
    template = load_template(name)
    if template is not None and revision is None:
        body = {k: v for k, v in template.items() if not k.startswith('_')}
        body.pop('name', None)
        digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:12]
        return IngestProfileRecord(
            name=name,
            source='template',
            read_only=True,
            revision=None,
            etag=f'ingest_profile:{name}:{digest}',
            description=TEMPLATE_DESCRIPTION,
            body=body,
        )
    return None


async def get_revision_record(name: str, revision: int) -> IngestProfileRecord | None:
    stored = await documents.get_revision(kind='ingest_profile', name=name, revision=revision)
    if stored is None:
        return None
    return _stored_record(
        name,
        revision=stored.revision,
        body=stored.body,
        src={
            'description': stored.description,
            'created_at': stored.created_at,
            'updated_at': stored.updated_at,
            'cloned_from': stored.cloned_from,
        },
        read_only=True,
    )


async def list_revisions(name: str) -> list[RevisionSummary] | None:
    if name not in get_config_store().current.ingest_profiles:
        return None
    infos = await documents.list_revisions(kind='ingest_profile', name=name)
    return [
        RevisionSummary(
            revision=i.revision,
            saved_at=i.created_at.isoformat(),
            cloned_from=i.cloned_from,
            description=i.description,
        )
        for i in infos or []
    ]


async def save_profile(
    *,
    name: str,
    body: dict[str, Any],
    expected_revision: int | None,
    description: str = '',
    cloned_from: CloneSource | None = None,
) -> IngestProfileRecord:
    await save_and_apply(
        'ingest_profile',
        name=name,
        body=body,
        expected_revision=expected_revision,
        description=description,
        cloned_from=cloned_from,
    )
    record = build_record(name)
    assert record is not None
    return record


async def delete_profile(*, name: str, expected_revision: int) -> None:
    await delete_and_apply('ingest_profile', name=name, expected_revision=expected_revision)


async def activate_profile(
    *, name: str | None, revision: int | None, expected_active: dict[str, Any] | None
) -> Activated:
    from src.services.config_store.store import activate_axis

    return await activate_axis(
        get_config_store(),
        axis='ingest_profile',
        name=name,
        revision=revision,
        expected_active=expected_active,
    )


async def rollback_profile(*, expected_active: dict[str, Any] | None) -> Activated:
    from src.services.config_store.activation_apply import rollback_axis

    return await rollback_axis(
        get_config_store(), axis='ingest_profile', expected_active=expected_active
    )


__all__ = [
    'ActiveConflictError',
    'IngestProfileRecord',
    'RevisionConflictError',
    'activate_profile',
    'all_known_names',
    'build_record',
    'delete_profile',
    'get_revision_record',
    'list_revisions',
    'load_template',
    'rollback_profile',
    'save_profile',
    'template_names',
]
