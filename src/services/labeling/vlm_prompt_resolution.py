"""Prompt-pack *resolution*: default/active pack selection, file-override
loading, and the config-store integration (activation pinning, §3.6/§3.7).

Split out of ``vlm_prompts.py`` (which stays domain-shape-only: the
``PromptPack`` dataclass and the built-in example packs) to stay under the
repo's 700-LOC pre-commit ratchet. ``vlm_prompts.py`` re-exports every name
here so every existing ``from src.services.labeling.vlm_prompts import
...`` call site keeps working unchanged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.core.logging import get_logger
from src.services.labeling.vlm_prompts import _BUILT_IN_NAMES, BUILT_IN_PACKS, PromptPack


logger = get_logger(__name__)

_PACK_FILE_CACHE: dict[str, tuple[int, PromptPack]] = {}


def _load_pack_file(path: Path) -> PromptPack | None:
    """Load a pack file, cached on ``(path, mtime)``; ``None`` (with a
    logged warning) when the file is missing or malformed."""
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        logger.warning('prompt_pack_path_missing', path=str(path))
        return None
    cached = _PACK_FILE_CACHE.get(str(path))
    if cached is not None and cached[0] == mtime:
        return cached[1]
    try:
        pack = PromptPack.from_json(path)
    except Exception as exc:
        logger.warning('prompt_pack_load_failed', path=str(path), error=str(exc))
        return None
    _PACK_FILE_CACHE[str(path)] = (mtime, pack)
    return pack


def _config(cfg: Any | None) -> Any:
    if cfg is None:
        from src.config.curation import get_curation_config

        return get_curation_config()
    return cfg


def resolve_prompt_pack(cfg: Any | None = None) -> PromptPack:
    """Resolve the *default* :class:`PromptPack` for this process.

    Mirrors the ``CurationConfig``-driven resolution
    ``get_curation_config()`` establishes for index names / paths (see
    ``docs/design/curation_design_rationale.md`` §2.1): a deployment
    points ``OP_PROMPT_PACK_PATH`` at its own JSON pack (pallets, food
    items, ...) instead of forking any code. Never raises -- a missing
    path, a missing file, or a malformed/incomplete pack all fall back to
    :data:`GENERIC_ITEM_PACK` with a logged warning, so a bad deployment
    config degrades the labeling vocabulary rather than crashing the
    process. Additional selectable packs (``OP_PROMPT_PACK_PATHS``) are
    listed by :func:`available_prompt_packs`.

    Args:
        cfg: A :class:`~src.config.curation.CurationConfig` instance, or
            ``None`` to use the process-wide default
            (``get_curation_config()``).
    """
    from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK

    path = getattr(_config(cfg), 'prompt_pack_path', None)
    if path is None:
        return GENERIC_ITEM_PACK
    pack = _load_pack_file(Path(path))
    return pack if pack is not None else GENERIC_ITEM_PACK


def available_prompt_packs(cfg: Any | None = None) -> dict[str, PromptPack]:
    """Every selectable pack, keyed by ``name``.

    Always includes the built-in packs (:data:`GENERIC_ITEM_PACK`, and the
    text-free :data:`GENERIC_REGION_PACK`), plus each
    loadable file in ``OP_PROMPT_PACK_PATHS``, the default
    ``OP_PROMPT_PACK_PATH`` pack, and every stored pack in the bound
    project's config-store snapshot (W2; empty until W3 CRUD exists, or
    on any process whose snapshot hasn't refreshed yet -- callers that
    need the latest cross-process state should
    ``await store.ensure_fresh(...)`` first). Unloadable files are
    skipped with a logged warning (same degrade-not-crash contract as
    :func:`resolve_prompt_pack`). On a name collision: stored packs
    cannot collide by construction (name uniqueness is enforced at
    save-time, W3); a file/default pack colliding with a built-in name
    is skipped; otherwise the default pack wins, then the earlier
    ``OP_PROMPT_PACK_PATHS`` entry.
    """
    config = _config(cfg)
    packs: dict[str, PromptPack] = {p.name: p for p in BUILT_IN_PACKS}
    default = resolve_prompt_pack(config)
    for path in getattr(config, 'prompt_pack_paths', ()) or ():
        pack = _load_pack_file(Path(path))
        if pack is None:
            continue
        if pack.name in packs and pack.name not in _BUILT_IN_NAMES:
            logger.warning('prompt_pack_name_collision', name=pack.name, path=str(path))
            continue
        packs[pack.name] = pack
    packs[default.name] = default
    packs.update(_stored_packs())
    return packs


def _stored_packs() -> dict[str, PromptPack]:
    """The bound project's stored packs from the process-local config-store
    snapshot (no I/O -- this reads whatever the last ``refresh``/
    ``ensure_fresh`` cached). Malformed stored bodies are skipped with a
    logged warning rather than raised, same degrade-not-crash contract as
    the file-pack loaders."""
    try:
        from src.services.config_store import get_config_store
    except Exception:  # pragma: no cover - config_store always importable
        return {}
    snapshot = get_config_store().current
    packs: dict[str, PromptPack] = {}
    for name, stored in snapshot.packs.items():
        try:
            packs[name] = PromptPack.from_dict({**stored.body, 'name': name})
        except Exception as exc:
            logger.warning('stored_prompt_pack_invalid', name=name, error=str(exc))
    return packs


def active_prompt_pack(cfg: Any | None = None) -> PromptPack:
    """The config-store's *active* pack (§3.6) if one is activated and
    still present, else the env/file process default
    (:func:`resolve_prompt_pack`) -- unchanged behavior for a deployment
    that never activates anything through the store.
    """
    try:
        from src.services.config_store import get_config_store
    except Exception:  # pragma: no cover - config_store always importable
        return resolve_prompt_pack(cfg)
    snapshot = get_config_store().current
    ref = snapshot.active_pack
    # 'off' has no meaning for a required axis (the VLM always needs some
    # pack) -- treat it the same as "never activated": fall back to the
    # env/file default, same as clearing a legacy settings override did.
    if ref is None or ref == 'off':
        return resolve_prompt_pack(cfg)
    name, _revision = ref
    # B1 fix: serve the pinned-at-activation revision, not `<name>`'s
    # current doc -- a PUT must not go live until a re-activate.
    pinned = snapshot.active_pack_body
    if pinned is not None and pinned.name == name:
        try:
            return PromptPack.from_dict({**pinned.body, 'name': name})
        except Exception as exc:
            logger.warning('active_prompt_pack_pinned_body_invalid', name=name, error=str(exc))
    pack = available_prompt_packs(cfg).get(name)
    return pack if pack is not None else resolve_prompt_pack(cfg)


def get_prompt_pack(
    name: str, cfg: Any | None = None, *, revision: int | None = None
) -> PromptPack | None:
    """The selectable pack called ``name``, or ``None`` if not configured.

    ``revision=<N>``: the *current* stored revision if it matches, else
    the pinned *activated* revision's copy (a per-run ``name@rev`` pinning
    the still-active-but-superseded revision, §3.7) -- ``None`` (honest)
    for any other revision this process hasn't cached. ``revision=None``:
    the *pinned* body when ``name`` is the store's active pack (B1
    round-2 -- every "active pack by name" caller must see
    :func:`active_prompt_pack`'s body, never an un-activated current
    doc); any other name resolves to its current doc, unchanged.
    """
    if revision is not None:
        from src.services.config_store import get_config_store

        snapshot = get_config_store().current
        stored = snapshot.packs.get(name)
        if stored is None or stored.revision != revision:
            pinned = snapshot.active_pack_body
            matches = pinned is not None and pinned.name == name and pinned.revision == revision
            stored = pinned if matches else None
        if stored is None:
            return None
        try:
            return PromptPack.from_dict({**stored.body, 'name': name})
        except Exception as exc:
            logger.warning('get_prompt_pack_stored_body_invalid', name=name, error=str(exc))
            return None
    try:
        from src.services.config_store import get_config_store

        ref = get_config_store().current.active_pack
    except Exception:  # pragma: no cover - config_store always importable
        ref = None
    if isinstance(ref, tuple) and ref[0] == name:
        return active_prompt_pack(cfg)
    return available_prompt_packs(cfg).get(name)


def prompt_pack_stamp(pack: PromptPack) -> str:
    """``"<name>@<revision|sha12>"`` provenance stamp for ``vlm_prompt_pack``
    (any_domain_plan.md §3.7/§9 W2) -- every VLM write site stamps this
    onto the item it wrote so a later audit can tell which pack produced
    the write. When ``pack`` is the store's currently *activated* pack,
    the stamp uses its exact saved revision; otherwise (a file/built-in
    pack the store never activated) a content hash distinguishes two
    edits of the same name.
    """
    try:
        from src.services.config_store import get_config_store

        ref = get_config_store().current.active_pack
    except Exception:  # pragma: no cover - config_store always importable
        ref = None
    if isinstance(ref, tuple):
        name, revision = ref
        if name == pack.name and revision is not None:
            return f'{pack.name}@{revision}'
    digest = hashlib.sha256(json.dumps(pack.to_dict(), sort_keys=True).encode()).hexdigest()[:12]
    return f'{pack.name}@{digest}'


__all__ = [
    'active_prompt_pack',
    'available_prompt_packs',
    'get_prompt_pack',
    'prompt_pack_stamp',
    'resolve_prompt_pack',
]
