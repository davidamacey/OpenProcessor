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


async def create_registry_class(
    reg: ClassRegistry,
    client: Any,
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
            hotkey_letter,
            class_id=None,
            registry_obj=await reg.snapshot(client),
            keymap_overrides=keymap_overrides,
        )
    new_id = await reg.add_class(client, name, group=group, notes=notes, hotkey_letter=letter)
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
        return await create_registry_class(
            reg,
            opensearch,
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
        new_letter: str | None = None
        if payload.hotkey_letter is not None:
            new_letter = validated_hotkey(
                payload.hotkey_letter,
                class_id=class_id,
                registry_obj=await reg.snapshot(opensearch),
                keymap_overrides=await project_keymap_overrides(opensearch),
            )
        # Name, group and hotkey change in one versioned write: two PUTs to different
        # classes (or fields) never overwrite each other.
        entry = await reg.update_class(
            opensearch,
            class_id,
            name=payload.name,
            group=payload.group,
            hotkey_letter=new_letter,
            set_hotkey=payload.hotkey_letter is not None,
        )
    except HTTPException:
        raise
    except ClassRegistryError as exc:
        if 'not found' in str(exc):
            raise HTTPException(status_code=404, detail=f'class_id {class_id} not found') from exc
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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
    snap = await reg.snapshot(opensearch)
    entry = snap.get(class_id)
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
        return await reg.set_deprecated(opensearch, class_id, True)
    except ClassRegistryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post('/classes/{class_id}/restore', response_model=RegistryClassEntry)
async def restore_class(class_id: int, opensearch: OpenSearchDep) -> RegistryClassEntry:
    """Undo ``POST /classes/{id}/deprecate``. ``404`` for an unknown id;
    ``409`` if a non-deprecated class already uses this class's name
    (the registry's name-uniqueness rule, enforced the same way
    ``update_class`` enforces it on rename).

    A class merged via ``POST /classes/merge`` (``merged_into`` set) is
    NOT restorable this way — its crops were already bulk-relabeled onto
    the merge target, so flipping ``deprecated`` back to ``false`` would
    resurrect an empty class while the data stays on the target. ``409``
    with a structured detail naming the merge target and pointing at the
    manual-relabel path instead. A plainly deprecated class (no
    ``merged_into``) restores as before.
    """
    reg = get_class_registry()
    snap = await reg.snapshot(opensearch)
    entry = snap.get(class_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f'unknown class_id {class_id}')
    if entry.merged_into is not None:
        target = snap.get(entry.merged_into)
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
        return await reg.set_deprecated(opensearch, class_id, False)
    except ClassRegistryError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
