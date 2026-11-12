"""Open-vocabulary prompt-set CRUD (the ``open_vocab_set`` kind /
``open_vocab`` axis). Mirrors :mod:`src.services.config_store.profiles`
without the env/registered sources: a set is either stored in the project's
config index or a read-only shipped template (``examples/open_vocab/``).
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
    from src.services.detection.open_vocab_set import OpenVocabSet


_TEMPLATES_DIR = Path(__file__).resolve().parents[3] / 'examples' / 'open_vocab'
_KIND: Literal['open_vocab_set'] = 'open_vocab_set'


@dataclass(frozen=True)
class OpenVocabRecord(KindRecord):
    pass


def load_template(name: str) -> dict[str, Any] | None:
    return templates.load_template(_TEMPLATES_DIR, name)


def template_names() -> list[str]:
    return list(templates.template_names(_TEMPLATES_DIR))


def all_known_names() -> frozenset[str]:
    return frozenset({*get_config_store().current.open_vocab_sets, *template_names()})


def _active_ref() -> tuple[str | None, int | None]:
    return templates.active_ref(get_config_store().current.active_open_vocab)


def build_record(name: str, *, revision: int | None = None) -> OpenVocabRecord | None:
    return build_kind_record(
        OpenVocabRecord,
        etag_prefix='open_vocab',
        name=name,
        revision=revision,
        stored=get_config_store().current.open_vocab_sets.get(name),
        template=load_template(name),
        active_ref=_active_ref(),
    )


async def get_revision_record(name: str, revision: int) -> OpenVocabRecord | None:
    stored = await documents.get_revision(kind=_KIND, name=name, revision=revision)
    if stored is None:
        return None
    return stored_record(
        OpenVocabRecord,
        etag_prefix='open_vocab',
        stored=stored,
        name=name,
        active_ref=_active_ref(),
        read_only=True,
    )


async def list_revisions(name: str) -> list[RevisionSummary] | None:
    if name not in get_config_store().current.open_vocab_sets:
        return None
    infos = await documents.list_revisions(kind='open_vocab_set', name=name)
    return revision_summaries(infos)


async def save_set(
    *,
    name: str,
    body: dict[str, Any],
    expected_revision: int | None,
    description: str = '',
    cloned_from: CloneSource | None = None,
) -> OpenVocabRecord:
    await save_and_apply(
        'open_vocab_set',
        name=name,
        body=body,
        expected_revision=expected_revision,
        description=description,
        cloned_from=cloned_from,
    )
    record = build_record(name)
    assert record is not None
    return record


async def delete_set(*, name: str, expected_revision: int) -> None:
    await delete_and_apply('open_vocab_set', name=name, expected_revision=expected_revision)


async def activate_set(
    *, name: str | None, revision: int | None, expected_active: dict[str, Any] | None
) -> Activated:
    from src.services.config_store.store import activate_axis

    return await activate_axis(
        get_config_store(),
        axis='open_vocab',
        name=name,
        revision=revision,
        expected_active=expected_active,
    )


async def rollback_set(*, expected_active: dict[str, Any] | None) -> Activated:
    from src.services.config_store.activation_apply import rollback_axis

    return await rollback_axis(
        get_config_store(), axis='open_vocab', expected_active=expected_active
    )


def active_open_vocab_set() -> OpenVocabSet | None:
    """The bound project's ACTIVATED set (the pinned revision's body), or
    ``None`` when none is active or the pinned body does not decode. Reads
    the process snapshot: callers that must see another worker's activation
    refresh the store first."""
    from src.services.detection.open_vocab_set import decode_open_vocab_set

    pinned = get_config_store().current.active_open_vocab_body
    if pinned is None:
        return None
    try:
        return decode_open_vocab_set(pinned.name, pinned.body)
    except ValueError:
        return None


__all__ = [
    'ActiveConflictError',
    'OpenVocabRecord',
    'RevisionConflictError',
    'activate_set',
    'active_open_vocab_set',
    'all_known_names',
    'build_record',
    'delete_set',
    'get_revision_record',
    'list_revisions',
    'load_template',
    'rollback_set',
    'save_set',
    'template_names',
]
