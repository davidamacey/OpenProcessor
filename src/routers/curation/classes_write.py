"""Curation router sub-module — class registry writes: create, update, deprecate, restore."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, status

from src.clients.curation_opensearch import RegistryClassEntry
from src.clients.curation_opensearch.registry import ClassRegistry, ClassRegistryError
from src.routers.curation._class_hotkeys import project_keymap_overrides, validated_hotkey
from src.routers.curation._class_models import ClassCreateRequest, ClassUpdateRequest  # noqa: TC001 - FastAPI
from src.routers.curation._common import (
    OpenSearchDep,
    get_class_registry,
    items_index,
    labels_confirmed_index,
    router,
)


def create_registry_class(
    reg: ClassRegistry,
    *,
    name: str,
    group: str = 'unknown',
    notes: str = '',
    hotkey_letter: str | None = None,
    keymap_overrides: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Shared class-creation path: ``POST /classes`` and the new-class
    proposal resolve route (``POST /review/new_class_proposals/resolve``)
    both create through this one function so there is exactly one place
    that adds a class and writes its optional hotkey.

    ``hotkey_letter`` (if given) is validated the same way as on
    ``PUT /classes/{id}`` *before* anything is written, against the
    bound project's *stored* keymap overrides (``keymap_overrides``) --
    omitted only by the new-class-proposal resolve route, which has no
    ``hotkey_letter`` field on its request at all today, so this branch
    never runs there. Raises ``ClassRegistryError`` on a duplicate
    (non-deprecated) name — the caller maps that to ``409``.
    """
    letter = None
    if hotkey_letter is not None:
        letter = validated_hotkey(
            hotkey_letter, class_id=None, registry_obj=reg.load(), keymap_overrides=keymap_overrides
        )
    new_id = reg.add_class(name, group=group, notes=notes)
    if letter is not None:
        registry_obj = reg.load()
        for c in registry_obj.classes:
            if c.class_id == new_id:
                c.hotkey_letter = letter
        reg._atomic_write(registry_obj)
    return {
        'class_id': new_id,
        'class_name': name,
        'group': group,
        'hotkey_letter': letter,
    }


@router.post('/classes', status_code=status.HTTP_201_CREATED)
async def create_class(payload: ClassCreateRequest, opensearch: OpenSearchDep) -> dict[str, Any]:
    """Append-only add. ``name`` must be a slug (``^[a-z0-9_]+$``, else 422);
    an optional ``hotkey_letter`` is validated like on update before
    anything is written, against the bound project's stored keymap."""
    reg = get_class_registry()
    keymap_overrides = (
        await project_keymap_overrides(opensearch) if payload.hotkey_letter is not None else None
    )
    try:
        return create_registry_class(
            reg,
            name=payload.name,
            group=payload.group,
            notes=payload.notes,
            hotkey_letter=payload.hotkey_letter,
            keymap_overrides=keymap_overrides,
        )
    except ClassRegistryError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.put('/classes/{class_id}')
async def update_class(
    class_id: int, payload: ClassUpdateRequest, opensearch: OpenSearchDep
) -> dict[str, Any]:
    """Rename, regroup, or assign/clear a hotkey on a class."""
    reg = get_class_registry()
    try:
        if payload.name is not None:
            reg.rename_class(class_id, payload.name)
        # group + hotkey_letter share the load+mutate+write path.
        if payload.group is not None or payload.hotkey_letter is not None:
            registry_obj = reg.load()
            new_letter: str | None = None
            if payload.hotkey_letter is not None:
                keymap_overrides = await project_keymap_overrides(opensearch)
                new_letter = validated_hotkey(
                    payload.hotkey_letter,
                    class_id=class_id,
                    registry_obj=registry_obj,
                    keymap_overrides=keymap_overrides,
                )
            for c in registry_obj.classes:
                if c.class_id == class_id:
                    if payload.group is not None:
                        c.group = payload.group
                    if payload.hotkey_letter is not None:
                        c.hotkey_letter = new_letter
                    break
            reg._atomic_write(registry_obj)
    except HTTPException:
        raise
    except ClassRegistryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    entry = reg.get(class_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f'class_id {class_id} not found')
    return entry.model_dump()


async def _count_class_item_references(opensearch: Any, class_id: int) -> int:
    """Items index docs whose ``class_id`` == this class. Same one-term
    ``count`` shape ``merge_class`` already uses for its holdout guard."""
    resp = await opensearch.count(
        index=items_index(), body={'query': {'term': {'class_id': class_id}}}
    )
    return int(resp.get('count', 0))


async def _count_class_confirmed_label_references(opensearch: Any, class_id: int) -> int:
    """Confirmed-labels index docs whose ``class_id`` == this class (the
    index ``merge_class`` bulk-relabels via ``update_by_query``)."""
    resp = await opensearch.count(
        index=labels_confirmed_index(), body={'query': {'term': {'class_id': class_id}}}
    )
    return int(resp.get('count', 0))


@router.post('/classes/{class_id}/deprecate', response_model=RegistryClassEntry)
async def deprecate_class(class_id: int, opensearch: OpenSearchDep) -> RegistryClassEntry:
    """Retire a class created by mistake that has no data yet.

    Unlike ``POST /classes/merge`` (which deprecates a source class while
    relabeling its items into a target), this needs no target — it only
    flips ``deprecated`` on a class nothing references. Refuses with
    ``409`` (naming the blocking counts) while any item or confirmed-label
    doc still carries this ``class_id`` — merge instead. ``404`` for an
    unknown id. Idempotent: calling this on an already-deprecated class
    just returns it (no reference re-check). Clears any bound
    ``hotkey_letter``.
    """
    reg = get_class_registry()
    entry = reg.get(class_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f'unknown class_id {class_id}')
    if entry.deprecated:
        return entry

    item_count = await _count_class_item_references(opensearch, class_id)
    label_count = await _count_class_confirmed_label_references(opensearch, class_id)
    if item_count or label_count:
        raise HTTPException(
            status_code=409,
            detail={
                'error': 'class_still_referenced',
                'message': (
                    f'class_id {class_id} is still referenced by data; merge it into '
                    'another class instead (POST /classes/merge)'
                ),
                'class_id': class_id,
                'item_count': item_count,
                'confirmed_label_count': label_count,
            },
        )
    try:
        return reg.set_deprecated(class_id, True)
    except ClassRegistryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post('/classes/{class_id}/restore', response_model=RegistryClassEntry)
async def restore_class(class_id: int) -> RegistryClassEntry:
    """Undo ``POST /classes/{id}/deprecate``. ``404`` for an unknown id;
    ``409`` if a non-deprecated class already uses this class's name
    (the registry's name-uniqueness rule, enforced the same way
    ``rename_class`` enforces it).

    A class merged via ``POST /classes/merge`` (``merged_into`` set) is
    NOT restorable this way — its crops were already bulk-relabeled onto
    the merge target, so flipping ``deprecated`` back to ``false`` would
    resurrect an empty class while the data stays on the target. ``409``
    with a structured detail naming the merge target and pointing at the
    manual-relabel path instead. A plainly deprecated class (no
    ``merged_into``) restores as before.
    """
    reg = get_class_registry()
    entry = reg.get(class_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f'unknown class_id {class_id}')
    if entry.merged_into is not None:
        target = reg.get(entry.merged_into)
        raise HTTPException(
            status_code=409,
            detail={
                'error': 'class_merged',
                'message': (
                    f'class_id {class_id} was merged into class_id {entry.merged_into} and '
                    'cannot be restored directly; its crops were already relabeled onto the '
                    'merge target'
                ),
                'class_id': class_id,
                'merged_into': {
                    'class_id': entry.merged_into,
                    'class_name': target.class_name if target is not None else None,
                },
                'hint': (
                    'relabel the crops you want back on this class manually '
                    '(PUT /crops/{id}/label or /crops/batch_label) rather than restoring it'
                ),
            },
        )
    try:
        return reg.set_deprecated(class_id, False)
    except ClassRegistryError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
