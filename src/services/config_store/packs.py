"""Prompt-pack CRUD service functions.

Thin wrappers over :mod:`src.services.config_store.documents`'s OCC
primitives, specialized to the ``prompt_pack`` kind/axis and to the
pack-specific sources (builtin, file, template, stored) -- kept separate
from the low-level module so it stays dependency-light (any_domain_plan.md
§3.6).
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
from src.services.labeling.region_overlay import REPLY_TEXT_KEY
from src.services.labeling.vlm_prompts import BUILT_IN_PACKS, PromptPack


if TYPE_CHECKING:
    from src.services.config_store.documents import Activated

PackSource = Literal['builtin', 'file', 'template', 'stored']

_TEMPLATES_DIR = Path(__file__).resolve().parents[3] / 'examples' / 'prompt_packs'


@dataclass(frozen=True)
class PackRecord:
    name: str
    source: PackSource
    read_only: bool
    revision: int | None
    etag: str
    description: str
    body: dict[str, Any]
    created_at: str | None = None
    updated_at: str | None = None
    updated_by: str | None = None
    cloned_from: str | None = None
    active: bool = False
    active_revision: int | None = None
    asks_region_text: bool = False


@dataclass(frozen=True)
class RevisionSummary:
    revision: int
    saved_at: str | None
    cloned_from: str | None
    description: str


def _content_etag(name: str, body: dict[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:12]
    return f'prompt_pack:{name}:{digest}'


def _asks_region_text(body: dict[str, Any]) -> bool:
    for f in ('combined_user_template', 'combined_batch_rules', 'region_user', 'region_batch_user'):
        text = str(body.get(f) or '')
        if REPLY_TEXT_KEY in text or '"text"' in text:
            return True
    return False


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


def builtin_and_file_bodies() -> dict[str, PromptPack]:
    """Every non-stored selectable pack (builtin + file/default), keyed by name."""
    from src.services.labeling.vlm_prompts import available_prompt_packs

    all_packs = available_prompt_packs()
    store = get_config_store()
    stored_names = set(store.current.packs)
    return {name: pack for name, pack in all_packs.items() if name not in stored_names}


def all_known_names() -> frozenset[str]:
    """Every name currently taken across every source (builtin, file,
    stored, template) -- the uniqueness set create/clone/validate check
    against (any_domain_plan.md §3.2)."""
    store = get_config_store()
    return frozenset({*builtin_and_file_bodies(), *store.current.packs, *_template_names()})


def _active_ref() -> tuple[str | None, int | None]:
    store = get_config_store()
    ref = store.current.active_pack
    if ref is None or ref == 'off':
        return None, None
    return ref


def build_record(name: str, *, revision: int | None = None) -> PackRecord | None:
    """The current (or, for a stored pack, a specific past revision)
    record for ``name``, or ``None`` if it does not exist."""
    store = get_config_store()
    active_name, active_revision = _active_ref()
    stored = store.current.packs.get(name)
    if stored is not None and (revision is None or revision == stored.revision):
        return PackRecord(
            name=name,
            source='stored',
            read_only=False,
            revision=stored.revision,
            etag=f'prompt_pack:{name}:{stored.revision}',
            description=stored.description,
            body=stored.body,
            created_at=stored.created_at,
            updated_at=stored.updated_at,
            cloned_from=stored.cloned_from,
            active=(active_name == name and active_revision == stored.revision),
            active_revision=active_revision if active_name == name else None,
            asks_region_text=_asks_region_text(stored.body),
        )
    bodies = builtin_and_file_bodies()
    if name in bodies:
        pack = bodies[name]
        body = pack.to_dict()
        body.pop('name', None)
        source: PackSource = 'builtin' if name in {p.name for p in BUILT_IN_PACKS} else 'file'
        return PackRecord(
            name=name,
            source=source,
            read_only=True,
            revision=None,
            etag=_content_etag(name, body),
            description='Built-in example' if source == 'builtin' else 'Deployment-configured pack',
            body=body,
            active=(active_name == name),
            active_revision=None,
            asks_region_text=_asks_region_text(body),
        )
    template = load_template(name)
    if template is not None and revision is None:
        body = {k: v for k, v in template.items() if not k.startswith('_')}
        body.pop('name', None)
        return PackRecord(
            name=name,
            source='template',
            read_only=True,
            revision=None,
            etag=_content_etag(name, body),
            description=TEMPLATE_DESCRIPTION,
            body=body,
            active=False,
            active_revision=None,
            asks_region_text=_asks_region_text(body),
        )
    return None


async def get_revision_record(name: str, revision: int) -> PackRecord | None:
    stored = await documents.get_revision(kind='prompt_pack', name=name, revision=revision)
    if stored is None:
        return None
    active_name, active_revision = _active_ref()
    return PackRecord(
        name=name,
        source='stored',
        read_only=True,
        revision=stored.revision,
        etag=f'prompt_pack:{name}:{stored.revision}',
        description=stored.description,
        body=stored.body,
        created_at=stored.created_at,
        updated_at=stored.updated_at,
        cloned_from=stored.cloned_from,
        active=(active_name == name and active_revision == stored.revision),
        active_revision=active_revision if active_name == name else None,
        asks_region_text=_asks_region_text(stored.body),
    )


async def list_revisions(name: str) -> list[RevisionSummary] | None:
    if name not in get_config_store().current.packs:
        return None
    infos = await documents.list_revisions(kind='prompt_pack', name=name)
    return [
        RevisionSummary(
            revision=i.revision,
            saved_at=i.created_at.isoformat(),
            cloned_from=i.cloned_from,
            description=i.description,
        )
        for i in infos or []
    ]


async def save_pack(
    *,
    name: str,
    body: dict[str, Any],
    expected_revision: int | None,
    description: str = '',
    cloned_from: CloneSource | None = None,
) -> PackRecord:
    await save_and_apply(
        'prompt_pack',
        name=name,
        body=body,
        expected_revision=expected_revision,
        description=description,
        cloned_from=cloned_from,
    )
    record = build_record(name)
    assert record is not None
    return record


async def delete_pack(*, name: str, expected_revision: int) -> None:
    await delete_and_apply('prompt_pack', name=name, expected_revision=expected_revision)


async def activate_pack(
    *, name: str | None, revision: int | None, expected_active: dict[str, Any] | None
) -> Activated:
    """Activate a stored pack, or a built-in / file pack by name (``revision`` ``None``): the
    latter is stored as a system revision first."""
    from src.services.config_store.store import activate_axis

    return await activate_axis(
        get_config_store(),
        axis='prompt_pack',
        name=name,
        revision=revision,
        expected_active=expected_active,
    )


async def rollback_pack(*, expected_active: dict[str, Any] | None) -> Activated:
    from src.services.config_store.activation_apply import rollback_axis

    return await rollback_axis(
        get_config_store(), axis='prompt_pack', expected_active=expected_active
    )


__all__ = [
    'ActiveConflictError',
    'PackRecord',
    'RevisionConflictError',
    'RevisionSummary',
    'activate_pack',
    'all_known_names',
    'build_record',
    'builtin_and_file_bodies',
    'delete_pack',
    'get_revision_record',
    'list_revisions',
    'load_template',
    'rollback_pack',
    'save_pack',
]
