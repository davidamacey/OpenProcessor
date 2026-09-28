"""Region-profile CRUD service functions (W4, any_domain_plan.md §4).

Mirrors ``src.services.config_store.packs`` for the ``region_profile``
kind / ``detection_profile`` axis -- see that module's docstring for the
shared design rationale.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from src.services.config_store import get_config_store
from src.services.config_store.index import (
    ActiveConflictError,
    RevisionConflictError,
    config_doc_id,
    delete_config as _delete_config,
    get_config_revision,
    save_config as _save_config,
)
from src.services.config_store.store import StoredConfig


if TYPE_CHECKING:
    from src.config import DetectionProfile

ProfileSource = Literal['env', 'registered', 'stored', 'template']

_TEMPLATES_DIR = Path(__file__).resolve().parents[3] / 'examples' / 'region_profiles'


@dataclass(frozen=True)
class ProfileRecord:
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


@dataclass(frozen=True)
class RevisionSummary:
    revision: int
    saved_at: str | None
    cloned_from: str | None
    description: str


def _content_etag(name: str, body: dict[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, default=list).encode()).hexdigest()[
        :12
    ]
    return f'region_profile:{name}:{digest}'


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


def _profile_to_wire(profile: DetectionProfile) -> dict[str, Any]:
    """A :class:`DetectionProfile`'s fields (minus ``name``), JSON-typed:
    tuples/frozensets become (sorted, for frozensets) lists."""
    from dataclasses import asdict

    raw = asdict(profile)
    raw.pop('name', None)
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, frozenset):
            out[key] = sorted(value)
        elif isinstance(value, tuple):
            out[key] = list(value)
        else:
            out[key] = value
    return out


def _registered_bodies() -> dict[str, DetectionProfile]:
    """Every env/registered profile (not stored) -- the non-config-store
    sources ``get_profiles()`` already unions with the store's snapshot."""
    from src.services.detection.profile_registry import get_profiles

    store = get_config_store()
    stored_names = set(store.current.profiles)
    return {name: p for name, p in get_profiles().items() if name not in stored_names}


def all_known_names() -> frozenset[str]:
    store = get_config_store()
    return frozenset({*_registered_bodies(), *store.current.profiles, *_template_names()})


def _active_ref() -> tuple[str | None, int | None]:
    store = get_config_store()
    ref = store.current.active_profile
    if ref is None or ref == 'off':
        return None, None
    return ref


def build_record(name: str, *, revision: int | None = None) -> ProfileRecord | None:
    store = get_config_store()
    active_name, active_revision = _active_ref()
    stored = store.current.profiles.get(name)
    if stored is not None and (revision is None or revision == stored.revision):
        return ProfileRecord(
            name=name,
            source='stored',
            read_only=False,
            revision=stored.revision,
            etag=f'region_profile:{name}:{stored.revision}',
            description=stored.description,
            body=stored.body,
            created_at=stored.created_at,
            updated_at=stored.updated_at,
            cloned_from=stored.cloned_from,
            active=(active_name == name and active_revision == stored.revision),
            active_revision=active_revision if active_name == name else None,
        )
    registered = _registered_bodies()
    if name in registered and revision is None:
        body = _profile_to_wire(registered[name])
        return ProfileRecord(
            name=name,
            source='registered',
            read_only=True,
            revision=None,
            etag=_content_etag(name, body),
            description='Env/registered profile',
            body=body,
            active=(active_name == name),
            active_revision=None,
        )
    template = load_template(name)
    if template is not None and revision is None:
        body = {k: v for k, v in template.items() if not k.startswith('_')}
        body.pop('name', None)
        return ProfileRecord(
            name=name,
            source='template',
            read_only=True,
            revision=None,
            etag=_content_etag(name, body),
            description='Template (clone to use)',
            body=body,
            active=False,
            active_revision=None,
        )
    return None


async def get_revision_record(client: Any, name: str, revision: int) -> ProfileRecord | None:
    from opensearchpy.exceptions import NotFoundError

    store = get_config_store()
    try:
        doc = await client.get(
            index=store.index, id=config_doc_id('region_profile', name, revision)
        )
    except NotFoundError:
        return None
    src = doc['_source']
    active_name, active_revision = _active_ref()
    return ProfileRecord(
        name=name,
        source='stored',
        read_only=True,
        revision=int(src['revision']),
        etag=f'region_profile:{name}:{src["revision"]}',
        description=src.get('description') or '',
        body=src.get('body') or {},
        created_at=src.get('created_at'),
        updated_at=src.get('updated_at'),
        cloned_from=src.get('cloned_from'),
        active=(active_name == name and active_revision == int(src['revision'])),
        active_revision=active_revision if active_name == name else None,
    )


async def list_revisions(client: Any, name: str) -> list[RevisionSummary] | None:
    store = get_config_store()
    if name not in store.current.profiles:
        return None
    resp = await client.search(
        index=store.index,
        body={
            'size': 1000,
            'query': {
                'bool': {
                    'filter': [
                        {'term': {'doc_type': 'revision'}},
                        {'term': {'kind': 'region_profile'}},
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


async def save_profile(
    client: Any,
    *,
    name: str,
    body: dict[str, Any],
    expected_revision: int | None,
    description: str = '',
    cloned_from: str | None = None,
) -> ProfileRecord:
    store = get_config_store()
    doc = await _save_config(
        client,
        store.index,
        kind='region_profile',
        name=name,
        body=body,
        expected_revision=expected_revision,
        description=description,
        cloned_from=cloned_from,
    )
    new_revision = await get_config_revision(client, store.index)
    store.apply_local(
        config_revision=new_revision,
        profile=StoredConfig(
            kind='region_profile',
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


async def delete_profile(client: Any, *, name: str, expected_revision: int) -> None:
    store = get_config_store()
    await _delete_config(
        client, store.index, kind='region_profile', name=name, expected_revision=expected_revision
    )
    new_revision = await get_config_revision(client, store.index)
    store.apply_local(config_revision=new_revision, profile=None, name=name)


async def activate_profile(
    client: Any, *, name: str | None, revision: int | None, expected_active: dict[str, Any] | None
) -> dict[str, Any]:
    from src.services.config_store.store import activate_axis

    store = get_config_store()
    return await activate_axis(
        store,
        client,
        axis='detection_profile',
        name=name,
        revision=revision,
        expected_active=expected_active,
    )


async def rollback_profile(
    client: Any, *, expected_active: dict[str, Any] | None
) -> dict[str, Any]:
    # N4 fix (W3/W4 round-3 review): `rollback_axis` resolves the pinned
    # body BEFORE writing the new activation doc, so a transient error
    # aborts cleanly instead of surfacing a 500 after the write already
    # committed. Replaces the old write-then-resolve call into
    # ``index.rollback``.
    from src.services.config_store.activation_apply import rollback_axis

    store = get_config_store()
    return await rollback_axis(
        store, client, axis='detection_profile', expected_active=expected_active
    )


__all__ = [
    'ActiveConflictError',
    'ProfileRecord',
    'RevisionConflictError',
    'RevisionSummary',
    'activate_profile',
    'all_known_names',
    'build_record',
    'delete_profile',
    'get_revision_record',
    'list_revisions',
    'load_template',
    'rollback_profile',
    'save_profile',
]
