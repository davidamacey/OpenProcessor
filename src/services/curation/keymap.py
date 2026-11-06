"""Per-project configurable keymap.

The action registry (``src/config/keymap_actions.json``) is static data
shared by every project. What varies per project is the *override map*
stored in the project's ``project_keymaps`` row:
``{action_id: [combo, ...]}``. The effective keymap is defaults (+)
overrides.

The wire shape follows the configurable-keyboard-shortcuts wire proposal.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.core.logging import get_logger
from src.db.repos import config as config_repo, settings as settings_repo
from src.db.repos.config import RevisionConflictError as RepoRevisionConflictError
from src.db.session import async_session_factory, project_tx
from src.services.config_store.bound_project import bound_project_id


logger = get_logger(__name__)

_ACTIONS_JSON_PATH = Path(__file__).resolve().parents[2] / 'config' / 'keymap_actions.json'

_MODIFIER_ORDER = ('ctrl', 'meta', 'alt', 'shift')


@dataclass(frozen=True)
class KeymapContext:
    id: str
    label: str
    description: str
    includes: tuple[str, ...]
    class_hotkeys_live: bool


@dataclass(frozen=True)
class KeymapAction:
    id: str
    context: str
    group: str | None
    label: str
    description: str
    default: tuple[str, ...]
    modifiable: bool


@dataclass(frozen=True)
class KeymapGrammar:
    modifiers: tuple[str, ...]
    named_keys: tuple[str, ...]
    printable: str
    max_combos_per_action: int
    locked_keys: tuple[str, ...]
    browser_reserved: tuple[str, ...]


@dataclass(frozen=True)
class ActionRegistry:
    grammar: KeymapGrammar
    contexts: dict[str, KeymapContext]
    actions: dict[str, KeymapAction]
    raw: dict[str, Any] = field(repr=False)

    def active_set(self, context_id: str) -> set[str]:
        """``context_id`` plus the transitive closure of ``includes``."""
        seen: set[str] = set()
        stack = [context_id]
        while stack:
            cid = stack.pop()
            if cid in seen or cid not in self.contexts:
                continue
            seen.add(cid)
            stack.extend(self.contexts[cid].includes)
        return seen

    def locked_action_default(self, action: KeymapAction) -> set[str]:
        """The subset of ``action.default`` that is a grammar-locked key
        (owner rule: a locked action keeps its default; that default
        combo may never be taken by another action)."""
        return {c for c in action.default if c in self.grammar.locked_keys}


@functools.lru_cache(maxsize=1)
def load_registry() -> ActionRegistry:
    with _ACTIONS_JSON_PATH.open(encoding='utf-8') as f:
        raw = json.load(f)
    grammar = KeymapGrammar(
        modifiers=tuple(raw['grammar']['modifiers']),
        named_keys=tuple(raw['grammar']['named_keys']),
        printable=raw['grammar']['printable'],
        max_combos_per_action=raw['grammar']['max_combos_per_action'],
        locked_keys=tuple(raw['grammar']['locked_keys']),
        browser_reserved=tuple(raw['grammar']['browser_reserved']),
    )
    contexts = {
        c['id']: KeymapContext(
            id=c['id'],
            label=c['label'],
            description=c['description'],
            includes=tuple(c['includes']),
            class_hotkeys_live=c['class_hotkeys_live'],
        )
        for c in raw['contexts']
    }
    actions = {
        a['id']: KeymapAction(
            id=a['id'],
            context=a['context'],
            group=a.get('group'),
            label=a['label'],
            description=a['description'],
            default=tuple(a['default']),
            modifiable=a['modifiable'],
        )
        for a in raw['actions']
    }
    return ActionRegistry(grammar=grammar, contexts=contexts, actions=actions, raw=raw)


def region_context_ids() -> frozenset[str]:
    """Contexts whose actions are ``available: false`` with no region
    profile on the bound project (glue G3: ``review.region.*`` and
    ``box_edit.*``)."""
    return frozenset({'review.region', 'box_edit'})


def effective_keys(action: KeymapAction, overrides: dict[str, list[str]]) -> list[str]:
    if action.id in overrides:
        return list(overrides[action.id])
    return list(action.default)


def effective_map(overrides: dict[str, list[str]]) -> dict[str, list[str]]:
    registry = load_registry()
    return {aid: effective_keys(a, overrides) for aid, a in registry.actions.items()}


def is_single_char(combo: str) -> str | None:
    """The bare letter if ``combo`` is an unmodified single character,
    else ``None`` (CW-K §3.4: only unmodified single-character combos
    count toward ``reserved_hotkeys``)."""
    if len(combo) == 1:
        return combo
    return None


def reserved_hotkeys(overrides: dict[str, list[str]]) -> list[str]:
    """CW-K §3.4: single unmodified characters bound in any
    ``class_hotkeys_live`` context, across the *effective* keymap."""
    registry = load_registry()
    out: set[str] = set()
    for action in registry.actions.values():
        if not registry.contexts[action.context].class_hotkeys_live:
            continue
        for combo in effective_keys(action, overrides):
            letter = is_single_char(combo)
            if letter is not None:
                out.add(letter)
    return sorted(out)


def combo_grammar_error(combo: str) -> str | None:
    """``None`` if ``combo`` matches the wire grammar, else a short
    reason (CW-K §2.2: ``[ctrl+][meta+][alt+][shift+]<key>``, in exactly
    that order, no repeated modifier). Canonicalization matters:
    ``shift+ctrl+z`` must be rejected rather than silently accepted as a
    distinct combo from ``ctrl+shift+z`` -- otherwise it's a dead
    binding (the client's ``normalize()`` never emits it), it evades
    collision detection, and it evades the browser-reserved check
    (M2)."""
    if not combo:
        return 'empty combo'
    parts = combo.split('+')
    key = parts[-1]
    mods = parts[:-1]
    if len(mods) != len(set(mods)):
        return f'{combo!r} repeats a modifier'
    unknown_mods = [m for m in mods if m not in _MODIFIER_ORDER]
    if unknown_mods:
        return f'{combo!r} has an unrecognized modifier {unknown_mods[0]!r}'
    canonical_mods = [m for m in _MODIFIER_ORDER if m in mods]
    if canonical_mods != mods:
        return f'{combo!r} modifiers must be in ctrl+meta+alt+shift order'
    grammar = load_registry().grammar
    if key in grammar.named_keys:
        return None
    if len(key) == 1 and key in grammar.printable:
        return None
    return f'{combo!r} is not a recognized key'


# --------------------------------------------------------------------- store


@dataclass(frozen=True)
class KeymapDoc:
    overrides: dict[str, list[str]]
    revision: int
    updated_at: str | None
    is_default: bool


RevisionConflictError = RepoRevisionConflictError


async def get_keymap_doc() -> KeymapDoc:
    """The bound project's override map; revision ``0`` and the defaults for a project with no row."""
    project_id = await bound_project_id()
    async with async_session_factory()() as session, project_tx(session, project_id):
        row = await settings_repo.get_keymap(session, project_id)
    if row is None:
        return KeymapDoc(overrides={}, revision=0, updated_at=None, is_default=True)
    return KeymapDoc(
        overrides=row.overrides,
        revision=row.revision,
        updated_at=row.updated_at.isoformat(),
        is_default=not row.overrides,
    )


async def publish_keymap_changed(doc: KeymapDoc) -> None:
    """The ``config.changed`` event of a keymap write, carrying both the keymap's own revision and
    the project's ``config_epoch`` the write bumped."""
    from src.services.events.publish import apublish

    project_id = await bound_project_id()
    async with async_session_factory()() as session, project_tx(session, project_id):
        epoch = await config_repo.get_epoch(session, project_id)
    await apublish(
        'config.changed',
        {
            'topic': 'config',
            'axis': 'keymap',
            'name': None,
            'config_revision': epoch,
            'keymap_revision': doc.revision,
        },
    )


async def save_keymap_doc(*, overrides: dict[str, list[str]], expected_revision: int) -> KeymapDoc:
    """OCC write of the whole override map. An override equal to the action's default is dropped so
    ``is_default`` stays honest (CW-K §4.3). A stale ``expected_revision`` is
    :class:`RevisionConflictError`."""
    registry = load_registry()
    cleaned = {
        aid: combos
        for aid, combos in overrides.items()
        if aid in registry.actions and list(combos) != list(registry.actions[aid].default)
    }
    project_id = await bound_project_id()
    async with async_session_factory()() as session, project_tx(session, project_id):
        row = await settings_repo.put_keymap(
            session, project_id, overrides=cleaned, expected_revision=expected_revision
        )
    return KeymapDoc(
        overrides=cleaned,
        revision=row.revision,
        updated_at=row.updated_at.isoformat(),
        is_default=not cleaned,
    )


__all__ = [
    'ActionRegistry',
    'KeymapAction',
    'KeymapContext',
    'KeymapDoc',
    'KeymapGrammar',
    'RevisionConflictError',
    'combo_grammar_error',
    'effective_keys',
    'effective_map',
    'get_keymap_doc',
    'is_single_char',
    'load_registry',
    'region_context_ids',
    'reserved_hotkeys',
    'save_keymap_doc',
]
