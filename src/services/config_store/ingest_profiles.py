"""Ingest detector profile CRUD (the ``ingest_profile`` kind / axis). Mirrors
:mod:`src.services.config_store.open_vocab`: a profile is either stored in the
project's config index or a read-only shipped template
(``examples/ingest_profiles/``). The deployment's ``OP_INGEST_*`` profile is the
default a project with no active profile runs; it is not a stored profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from src.services.config_store import documents, get_config_store, templates
from src.services.config_store.documents import (
    ActiveConflictError,
    CloneSource,
    RevisionConflictError,
)
from src.services.config_store.kind_records import (
    KindRecord,
    build_kind_record,
    revision_summaries,
    stored_record,
)
from src.services.config_store.stored_ops import delete_and_apply, save_and_apply


if TYPE_CHECKING:
    from src.services.config_store.documents import Activated
    from src.services.config_store.profiles import RevisionSummary

ProfileSource = Literal['stored', 'template']

_TEMPLATES_DIR = Path(__file__).resolve().parents[3] / 'examples' / 'ingest_profiles'
_KIND: Literal['ingest_profile'] = 'ingest_profile'


@dataclass(frozen=True)
class IngestProfileRecord(KindRecord):
    pass


def load_template(name: str) -> dict[str, Any] | None:
    return templates.load_template(_TEMPLATES_DIR, name)


def template_names() -> list[str]:
    return list(templates.template_names(_TEMPLATES_DIR))


def all_known_names() -> frozenset[str]:
    return frozenset({*get_config_store().current.ingest_profiles, *template_names()})


def _active_ref() -> tuple[str | None, int | None]:
    return templates.active_ref(get_config_store().current.active_ingest_profile)


def build_record(name: str, *, revision: int | None = None) -> IngestProfileRecord | None:
    return build_kind_record(
        IngestProfileRecord,
        etag_prefix='ingest_profile',
        name=name,
        revision=revision,
        stored=get_config_store().current.ingest_profiles.get(name),
        template=load_template(name),
        active_ref=_active_ref(),
    )


async def get_revision_record(name: str, revision: int) -> IngestProfileRecord | None:
    stored = await documents.get_revision(kind=_KIND, name=name, revision=revision)
    if stored is None:
        return None
    return stored_record(
        IngestProfileRecord,
        etag_prefix='ingest_profile',
        stored=stored,
        name=name,
        active_ref=_active_ref(),
        read_only=True,
    )


async def list_revisions(name: str) -> list[RevisionSummary] | None:
    if name not in get_config_store().current.ingest_profiles:
        return None
    infos = await documents.list_revisions(kind='ingest_profile', name=name)
    return revision_summaries(infos)


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
