"""Region-profile CRUD service functions.

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

from src.services.config_store import documents, get_config_store
from src.services.config_store.documents import (
    TEMPLATE_DESCRIPTION,
    ActiveConflictError,
    CloneSource,
    RevisionConflictError,
)
from src.services.config_store.stored_ops import delete_and_apply, save_and_apply


if TYPE_CHECKING:
    from src.config import DetectionProfile
    from src.services.config_store.documents import Activated

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
        typed: dict[str, Any] | None = json.loads(path.read_text(encoding='utf-8'))
        return typed
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
            description=TEMPLATE_DESCRIPTION,
            body=body,
            active=False,
            active_revision=None,
        )
    return None


async def get_revision_record(name: str, revision: int) -> ProfileRecord | None:
    stored = await documents.get_revision(kind='region_profile', name=name, revision=revision)
    if stored is None:
        return None
    active_name, active_revision = _active_ref()
    return ProfileRecord(
        name=name,
        source='stored',
        read_only=True,
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


async def list_revisions(name: str) -> list[RevisionSummary] | None:
    if name not in get_config_store().current.profiles:
        return None
    infos = await documents.list_revisions(kind='region_profile', name=name)
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
) -> ProfileRecord:
    await save_and_apply(
        'region_profile',
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
    await delete_and_apply('region_profile', name=name, expected_revision=expected_revision)


async def activate_profile(
    *, name: str | None, revision: int | None, expected_active: dict[str, Any] | None
) -> Activated:
    """Activate a stored profile, or an env / registered profile by name (``revision`` ``None``):
    the latter is stored as a system revision first."""
    from src.services.config_store.store import activate_axis

    return await activate_axis(
        get_config_store(),
        axis='detection_profile',
        name=name,
        revision=revision,
        expected_active=expected_active,
    )


async def rollback_profile(*, expected_active: dict[str, Any] | None) -> Activated:
    from src.services.config_store.activation_apply import rollback_axis

    return await rollback_axis(
        get_config_store(), axis='detection_profile', expected_active=expected_active
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
