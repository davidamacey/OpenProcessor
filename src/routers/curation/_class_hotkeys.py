"""Class-hotkey validation against the bound project's keymap (W2b).

Split out of ``classes.py`` to stay under the repo's 700-LOC pre-commit
ratchet; ``classes.py`` imports these helpers directly.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from src.routers.curation._config_common_models import api_error


async def project_keymap_overrides(opensearch: Any) -> dict[str, list[str]]:
    from src.config import get_curation_config
    from src.services.curation.keymap import get_keymap_doc

    cfg = get_curation_config()
    doc = await get_keymap_doc(opensearch, cfg.configs_index)
    return doc.overrides


async def project_reserved_hotkeys(opensearch: Any) -> list[str]:
    """``reserved_hotkeys`` derived from the bound project's *stored*
    keymap overrides (CW-K §3.4), not just the defaults."""
    from src.config import get_curation_config
    from src.services.curation.keymap import get_keymap_doc, reserved_hotkeys

    cfg = get_curation_config()
    doc = await get_keymap_doc(opensearch, cfg.configs_index)
    return reserved_hotkeys(doc.overrides)


def reserved_hotkey_actions(
    letter: str, keymap_overrides: dict[str, list[str]] | None = None
) -> list[dict[str, Any]]:
    """Every action whose *effective* key set (M1: the project's stored
    keymap overrides, not just the defaults -- the reservation itself
    comes from those overrides, so the explanation must too) binds this
    single character in a ``class_hotkeys_live`` context (CW-K §3.3)."""
    from src.services.curation.keymap import effective_keys, is_single_char, load_registry

    registry = load_registry()
    overrides = keymap_overrides or {}
    out = []
    for action in registry.actions.values():
        if not registry.contexts[action.context].class_hotkeys_live:
            continue
        for combo in effective_keys(action, overrides):
            if is_single_char(combo) == letter:
                out.append(
                    {'action_id': action.id, 'context': action.context, 'label': action.label}
                )
                break
    return out


def validated_hotkey(
    raw: str,
    *,
    class_id: int | None,
    registry_obj: Any,
    keymap_overrides: dict[str, Any] | None = None,
) -> str | None:
    """Normalize a requested hotkey; ``None`` = clear. 400 not one char,
    422 ``hotkey_reserved`` (a keymap action's key, CW-K §3.3), 409
    ``hotkey_taken`` (bound to another active class). ``keymap_overrides``
    is the bound project's *stored* keymap overrides -- every route that
    can actually write a hotkey (``POST/PUT /classes``) fetches and
    passes it; ``None`` falls back to the default keymap's reserved set,
    which only ever matters for the new-class-proposal resolve route
    (``review_resolve.py``), whose request has no ``hotkey_letter`` field
    at all today, so this branch never runs for it."""
    from src.services.curation.keymap import reserved_hotkeys

    stripped = raw.strip()
    if stripped == '':
        return None
    if len(stripped) != 1:
        raise HTTPException(status_code=400, detail='hotkey_letter must be a single character')
    letter = stripped.lower()
    if letter in reserved_hotkeys(keymap_overrides or {}):
        raise api_error(
            422,
            'hotkey_reserved',
            f"'{letter}' is reserved by the active keymap.",
            actions=reserved_hotkey_actions(letter, keymap_overrides),
        )
    for c in registry_obj.classes:
        if c.deprecated or c.class_id == class_id:
            continue
        if (getattr(c, 'hotkey_letter', None) or '').lower() == letter:
            raise api_error(
                409,
                'hotkey_taken',
                f"'{letter}' is already bound to '{c.class_name}'.",
                class_id=c.class_id,
                class_name=c.class_name,
            )
    return letter


__all__ = [
    'project_keymap_overrides',
    'project_reserved_hotkeys',
    'reserved_hotkey_actions',
    'validated_hotkey',
]
