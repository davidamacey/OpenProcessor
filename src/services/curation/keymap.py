"""W2b: per-project configurable keymap.

The action registry (``src/config/keymap_actions.json``) is static data
shared by every project. What varies per project is the *override map*
stored on the project's ``configs`` index (doc id ``keymap:default``):
``{action_id: [combo, ...]}``. The effective keymap is defaults (+)
overrides.

See ``docs/design/openprocessor_internal/any_domain_plan.md`` W2b and the
configurable-keyboard-shortcuts wire proposal (CW-K) §0 (binding), §2, §3.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from opensearchpy.exceptions import ConflictError, NotFoundError

from src.core.logging import get_logger


logger = get_logger(__name__)

_ACTIONS_JSON_PATH = Path(__file__).resolve().parents[2] / 'config' / 'keymap_actions.json'

KEYMAP_DOC_ID = 'keymap:default'

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

    def actions_in_context(self, context_id: str) -> list[KeymapAction]:
        active = self.active_set(context_id)
        return [a for a in self.actions.values() if a.context in active]

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


class RevisionConflictError(Exception):
    def __init__(self, current_revision: int) -> None:
        self.current_revision = current_revision
        super().__init__(f'keymap revision conflict (current={current_revision})')


async def get_keymap_doc(client: Any, index: str) -> KeymapDoc:
    try:
        doc = await client.get(index=index, id=KEYMAP_DOC_ID)
    except NotFoundError:
        return KeymapDoc(overrides={}, revision=0, updated_at=None, is_default=True)
    # Some lightweight test doubles answer a miss with ``{"found": False}``
    # instead of raising (real OpenSearch clients always raise).
    if isinstance(doc, dict) and doc.get('found') is False:
        return KeymapDoc(overrides={}, revision=0, updated_at=None, is_default=True)
    src = doc['_source']
    overrides = src.get('overrides') or {}
    return KeymapDoc(
        overrides=overrides,
        revision=int(src.get('revision', 0)),
        updated_at=src.get('updated_at'),
        is_default=not overrides,
    )


async def save_keymap_doc(
    client: Any, index: str, *, overrides: dict[str, list[str]], expected_revision: int
) -> KeymapDoc:
    """OCC write of the whole override map. An override equal to the
    action's default is dropped so ``is_default`` stays honest (CW-K
    §4.3)."""
    registry = load_registry()
    cleaned = {
        aid: combos
        for aid, combos in overrides.items()
        if aid in registry.actions and list(combos) != list(registry.actions[aid].default)
    }
    doc_id = KEYMAP_DOC_ID
    seq_no = primary_term = None
    current_revision = 0
    try:
        current = await client.get(index=index, id=doc_id)
        if not (isinstance(current, dict) and current.get('found') is False):
            current_revision = int(current['_source'].get('revision', 0))
            seq_no = current.get('_seq_no')
            primary_term = current.get('_primary_term')
    except NotFoundError:
        pass

    if expected_revision != current_revision:
        raise RevisionConflictError(current_revision)

    now = datetime.now(UTC).isoformat()
    next_revision = current_revision + 1
    doc = {
        'doc_type': 'keymap',
        'overrides': cleaned,
        'revision': next_revision,
        'updated_at': now,
    }
    index_kwargs: dict[str, Any] = {'index': index, 'id': doc_id, 'body': doc}
    if seq_no is not None:
        index_kwargs['if_seq_no'] = seq_no
        index_kwargs['if_primary_term'] = primary_term
    try:
        await client.index(**index_kwargs)
    except ConflictError as exc:
        # Re-read for the real current revision rather than reporting a
        # meaningless -1 -- a concurrent writer landed between our GET
        # above and this index() call.
        try:
            latest = await client.get(index=index, id=doc_id)
            real_current = int(latest['_source'].get('revision', 0))
        except Exception:
            real_current = -1
        raise RevisionConflictError(real_current) from exc

    # W2's atomic bump_config_revision (painless script + retry_on_conflict)
    # -- not a second read-then-write, so a concurrent config write can't
    # lose an increment (M6).
    from src.services.config_store.index import bump_config_revision

    await bump_config_revision(client, index)
    return KeymapDoc(
        overrides=cleaned, revision=next_revision, updated_at=now, is_default=not cleaned
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
