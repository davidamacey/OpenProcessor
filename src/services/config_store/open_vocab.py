"""Open-vocabulary prompt-set CRUD (the ``open_vocab_set`` kind /
``open_vocab`` axis). Mirrors :mod:`src.services.config_store.profiles`
without the env/registered sources: a set is either stored in the project's
config index or a read-only shipped template (``examples/open_vocab/``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from src.services.config_store import get_config_store
from src.services.config_store.index import (
    TEMPLATE_DESCRIPTION,
    ActiveConflictError,
    RevisionConflictError,
    config_doc_id,
    delete_config as _delete_config,
    get_config_revision,
    save_config as _save_config,
)
from src.services.config_store.profiles import RevisionSummary
from src.services.config_store.store import StoredConfig


if TYPE_CHECKING:
    from src.services.detection.open_vocab_set import OpenVocabSet

SetSource = Literal['stored', 'template']

_TEMPLATES_DIR = Path(__file__).resolve().parents[3] / 'examples' / 'open_vocab'
_KIND: Literal['open_vocab_set'] = 'open_vocab_set'


@dataclass(frozen=True)
class OpenVocabRecord:
    name: str
    source: SetSource
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
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def template_names() -> list[str]:
    return list(_template_names())


def all_known_names() -> frozenset[str]:
    return frozenset({*get_config_store().current.open_vocab_sets, *_template_names()})


def _active_ref() -> tuple[str | None, int | None]:
    ref = get_config_store().current.active_open_vocab
    if ref is None or ref == 'off':
        return None, None
    return ref


def _stored_record(
    name: str, *, revision: int, body: dict[str, Any], src: dict[str, Any], read_only: bool
) -> OpenVocabRecord:
    active_name, active_revision = _active_ref()
    return OpenVocabRecord(
        name=name,
        source='stored',
        read_only=read_only,
        revision=revision,
        etag=f'open_vocab:{name}:{revision}',
        description=src.get('description') or '',
        body=body,
        created_at=src.get('created_at'),
        updated_at=src.get('updated_at'),
        cloned_from=src.get('cloned_from'),
        active=(active_name == name and active_revision == revision),
        active_revision=active_revision if active_name == name else None,
    )


def build_record(name: str, *, revision: int | None = None) -> OpenVocabRecord | None:
    stored = get_config_store().current.open_vocab_sets.get(name)
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
        return OpenVocabRecord(
            name=name,
            source='template',
            read_only=True,
            revision=None,
            etag=f'open_vocab:{name}:{digest}',
            description=TEMPLATE_DESCRIPTION,
            body=body,
        )
    return None


async def get_revision_record(client: Any, name: str, revision: int) -> OpenVocabRecord | None:
    from opensearchpy.exceptions import NotFoundError

    try:
        doc = await client.get(
            index=get_config_store().index, id=config_doc_id(_KIND, name, revision)
        )
    except NotFoundError:
        return None
    src = doc['_source']
    return _stored_record(
        name,
        revision=int(src['revision']),
        body=src.get('body') or {},
        src=src,
        read_only=True,
    )


async def list_revisions(client: Any, name: str) -> list[RevisionSummary] | None:
    store = get_config_store()
    if name not in store.current.open_vocab_sets:
        return None
    resp = await client.search(
        index=store.index,
        body={
            'size': 1000,
            'query': {
                'bool': {
                    'filter': [
                        {'term': {'doc_type': 'revision'}},
                        {'term': {'kind': _KIND}},
                        {'term': {'name': name}},
                    ]
                }
            },
            'sort': [{'revision': {'order': 'desc'}}],
        },
    )
    return [
        RevisionSummary(
            revision=int(hit['_source']['revision']),
            saved_at=hit['_source'].get('updated_at'),
            cloned_from=hit['_source'].get('cloned_from'),
            description=hit['_source'].get('description') or '',
        )
        for hit in resp['hits']['hits']
    ]


async def save_set(
    client: Any,
    *,
    name: str,
    body: dict[str, Any],
    expected_revision: int | None,
    description: str = '',
    cloned_from: str | None = None,
) -> OpenVocabRecord:
    store = get_config_store()
    doc = await _save_config(
        client,
        store.index,
        kind=_KIND,
        name=name,
        body=body,
        expected_revision=expected_revision,
        description=description,
        cloned_from=cloned_from,
    )
    store.apply_local(
        config_revision=await get_config_revision(client, store.index),
        open_vocab_set=StoredConfig(
            kind=_KIND,
            name=name,
            revision=doc['revision'],
            body=doc['body'],
            description=doc['description'],
            created_at=doc['created_at'],
            updated_at=doc['updated_at'],
            cloned_from=doc.get('cloned_from'),
        ),
    )
    record = build_record(name)
    assert record is not None
    return record


async def delete_set(client: Any, *, name: str, expected_revision: int) -> None:
    store = get_config_store()
    await _delete_config(
        client, store.index, kind=_KIND, name=name, expected_revision=expected_revision
    )
    store.apply_local(
        config_revision=await get_config_revision(client, store.index),
        open_vocab_set=None,
        name=name,
    )


async def activate_set(
    client: Any, *, name: str | None, revision: int | None, expected_active: dict[str, Any] | None
) -> dict[str, Any]:
    from src.services.config_store.store import activate_axis

    return await activate_axis(
        get_config_store(),
        client,
        axis='open_vocab',
        name=name,
        revision=revision,
        expected_active=expected_active,
    )


async def rollback_set(client: Any, *, expected_active: dict[str, Any] | None) -> dict[str, Any]:
    from src.services.config_store.activation_apply import rollback_axis

    return await rollback_axis(
        get_config_store(), client, axis='open_vocab', expected_active=expected_active
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
