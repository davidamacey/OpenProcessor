"""``GET/PUT {prefix}/keymap``, ``POST {prefix}/keymap/validate``,
``POST {prefix}/keymap/reset`` -- the per-project configurable keymap
(W2b). Every route is project-scoped through the shared ``router`` --
there is no unscoped/global keymap route (owner, 2026-09-26).
"""

from __future__ import annotations

from typing import Any

from src.routers.curation._common import OpenSearchDep, get_class_registry, logger, router
from src.routers.curation._config_common_models import api_error
from src.routers.curation._keymap_models import (
    KeymapActionWire,
    KeymapContextWire,
    KeymapGetResponse,
    KeymapGrammarWire,
    KeymapPutRequest,
    KeymapPutResponse,
    KeymapResetRequest,
    KeymapValidateRequest,
    KeymapValidateResponse,
)
from src.services.curation.keymap import (
    KeymapDoc,
    RevisionConflictError,
    get_keymap_doc,
    load_registry,
    region_context_ids,
    reserved_hotkeys,
    save_keymap_doc,
)


# keymap_validator imports _config_common_models, which lives inside this
# same package -- importing it at module scope here would re-enter
# src.routers.curation.__init__ while it is still executing this very
# import (the package imports this module for its route side effects).
# Deferred to call sites instead.


def _classes_payload() -> list[dict[str, Any]]:
    reg = get_class_registry().load()
    return [
        {
            'class_id': c.class_id,
            'class_name': c.class_name,
            'hotkey_letter': c.hotkey_letter,
            'deprecated': c.deprecated,
        }
        for c in reg.classes
    ]


def _region_profile_available() -> bool:
    from src.services.config_store.store import get_config_store

    store = get_config_store()
    ref = store.current.active_profile
    return bool(ref) and ref != 'off'


def _build_response(
    doc: KeymapDoc, *, project: str, extra_issues: list[Any] | None = None
) -> KeymapGetResponse:
    registry = load_registry()
    has_region = _region_profile_available()
    region_ctx = region_context_ids()
    grammar = KeymapGrammarWire(
        modifiers=list(registry.grammar.modifiers),
        named_keys=list(registry.grammar.named_keys),
        printable=registry.grammar.printable,
        max_combos_per_action=registry.grammar.max_combos_per_action,
        locked_keys=list(registry.grammar.locked_keys),
        browser_reserved=list(registry.grammar.browser_reserved),
    )
    contexts = [
        KeymapContextWire(
            id=c.id,
            label=c.label,
            description=c.description,
            includes=list(c.includes),
            class_hotkeys_live=c.class_hotkeys_live,
        )
        for c in registry.contexts.values()
    ]
    actions = [
        KeymapActionWire(
            id=a.id,
            context=a.context,
            group=a.group,
            label=a.label,
            description=a.description,
            default=list(a.default),
            keys=doc.overrides.get(a.id, list(a.default)),
            modifiable=a.modifiable,
            available=has_region if a.context in region_ctx else True,
            locked_keys=sorted(registry.locked_action_default(a)),
        )
        for a in registry.actions.values()
    ]
    classes = _classes_payload()
    from src.services.curation.keymap_validator import shadowed_conflicts

    issues = list(extra_issues or []) or shadowed_conflicts(project, classes, doc.overrides)
    return KeymapGetResponse(
        project=project,
        revision=doc.revision,
        etag=f'keymap:{doc.revision}',
        is_default=doc.is_default,
        updated_at=doc.updated_at,
        grammar=grammar,
        contexts=contexts,
        actions=actions,
        overrides=doc.overrides,
        reserved_hotkeys=reserved_hotkeys(doc.overrides),
        issues=issues,
    )


@router.get('/keymap', response_model=KeymapGetResponse)
async def get_keymap(opensearch: OpenSearchDep) -> KeymapGetResponse:
    from src.config import get_curation_config

    cfg = get_curation_config()
    doc = await get_keymap_doc(opensearch, cfg.configs_index)
    return _build_response(doc, project=cfg.project_slug)


@router.post('/keymap/validate', response_model=KeymapValidateResponse)
async def validate_keymap_route(
    body: KeymapValidateRequest, opensearch: OpenSearchDep
) -> KeymapValidateResponse:
    from src.config import get_curation_config

    cfg = get_curation_config()
    current = await get_keymap_doc(opensearch, cfg.configs_index)
    merged = {**current.overrides, **body.overrides}
    classes = _classes_payload()
    from src.services.curation.keymap_validator import validate_keymap

    report, class_conflicts, resolved = validate_keymap(
        merged, project=cfg.project_slug, classes=classes, previous_overrides=current.overrides
    )
    return KeymapValidateResponse(
        ok=report.ok,
        errors=report.errors,
        warnings=report.warnings,
        force_allowed=report.force_allowed,
        resolved=resolved,
        reserved_hotkeys=reserved_hotkeys(merged),
        class_conflicts=[
            {
                'project': c.project,
                'class_id': c.class_id,
                'class_name': c.class_name,
                'combo': c.combo,
                'action_id': c.action_id,
            }
            for c in class_conflicts
        ],
    )


async def _unbind_classes(class_conflicts: list[Any]) -> list[dict[str, Any]]:
    """Clear each conflicting class's ``hotkey_letter`` in one operation
    (CW-K §3.2) and re-sync the registry. Returns the echoed
    ``unbound_class_hotkeys``."""
    if not class_conflicts:
        return []
    registry_obj = get_class_registry()
    reg = registry_obj.load()
    unbound: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for conflict in class_conflicts:
        if conflict.class_id in seen_ids:
            continue
        seen_ids.add(conflict.class_id)
        for c in reg.classes:
            if c.class_id == conflict.class_id:
                was = c.hotkey_letter
                c.hotkey_letter = None
                unbound.append(
                    {
                        'project': conflict.project,
                        'class_id': conflict.class_id,
                        'class_name': conflict.class_name,
                        'was': was,
                    }
                )
                break
    if unbound:
        registry_obj._atomic_write(reg)
        try:
            from src.services.curation.event_hub import get_event_hub

            get_event_hub().publish({'type': 'classes.changed', 'topic': 'classes'})
        except Exception as exc:  # pragma: no cover - advisory only
            logger.warning('classes_changed_publish_failed', error=str(exc))
    return unbound


async def _publish_keymap_changed(doc: KeymapDoc) -> None:
    from src.services.curation.event_hub import get_event_hub

    get_event_hub().publish(
        {
            'type': 'config.changed',
            'topic': 'config',
            'axis': 'keymap',
            'name': None,
            'keymap_revision': doc.revision,
        }
    )


@router.put('/keymap', response_model=KeymapPutResponse)
async def put_keymap(body: KeymapPutRequest, opensearch: OpenSearchDep) -> KeymapPutResponse:
    from src.config import get_curation_config

    cfg = get_curation_config()
    current = await get_keymap_doc(opensearch, cfg.configs_index)
    if body.expected_revision != current.revision:
        raise api_error(
            409,
            'revision_conflict',
            'The keymap changed since you loaded it.',
            current_revision=current.revision,
        )

    # PUT replaces the whole override map (CW-K §4.3) -- an action
    # absent from ``body.overrides`` takes its default, not its prior
    # override.
    merged = dict(body.overrides)

    classes = _classes_payload()
    from src.services.curation.keymap_validator import validate_keymap

    report, class_conflicts, _resolved = validate_keymap(
        merged, project=cfg.project_slug, classes=classes, previous_overrides=current.overrides
    )
    if not report.ok:
        raise api_error(
            422,
            'validation_failed',
            f'The keymap has {len(report.errors)} error(s).',
            current_revision=current.revision,
            report=report,
        )
    if class_conflicts and not body.unbind_conflicting_class_hotkeys:
        conflict = class_conflicts[0]
        raise api_error(
            409,
            'class_hotkey_conflict',
            f"'{conflict.combo}' is bound to class '{conflict.class_name}' in project "
            f'{conflict.project}.',
            current_revision=current.revision,
            class_conflicts=[
                {
                    'project': c.project,
                    'class_id': c.class_id,
                    'class_name': c.class_name,
                    'combo': c.combo,
                    'action_id': c.action_id,
                }
                for c in class_conflicts
            ],
        )

    unbound = (
        await _unbind_classes(class_conflicts) if body.unbind_conflicting_class_hotkeys else []
    )

    try:
        new_doc = await save_keymap_doc(
            opensearch,
            cfg.configs_index,
            overrides=merged,
            expected_revision=body.expected_revision,
        )
    except RevisionConflictError as exc:
        raise api_error(
            409,
            'revision_conflict',
            'The keymap changed since you loaded it.',
            current_revision=exc.current_revision,
        ) from exc

    await _publish_keymap_changed(new_doc)
    logger.info('keymap_updated', project=cfg.project_slug, revision=new_doc.revision)
    response = _build_response(new_doc, project=cfg.project_slug)
    return KeymapPutResponse(**response.model_dump(), unbound_class_hotkeys=unbound)


@router.post('/keymap/reset', response_model=KeymapPutResponse)
async def reset_keymap(body: KeymapResetRequest, opensearch: OpenSearchDep) -> KeymapPutResponse:
    from src.config import get_curation_config

    cfg = get_curation_config()
    current = await get_keymap_doc(opensearch, cfg.configs_index)
    if body.expected_revision != current.revision:
        raise api_error(
            409,
            'revision_conflict',
            'The keymap changed since you loaded it.',
            current_revision=current.revision,
        )

    if body.action_ids is None:
        merged: dict[str, list[str]] = {}
    else:
        merged = {
            aid: combos for aid, combos in current.overrides.items() if aid not in body.action_ids
        }

    classes = _classes_payload()
    from src.services.curation.keymap_validator import validate_keymap

    report, class_conflicts, _resolved = validate_keymap(
        merged, project=cfg.project_slug, classes=classes, previous_overrides=current.overrides
    )
    if not report.ok:
        raise api_error(
            422,
            'validation_failed',
            f'The keymap has {len(report.errors)} error(s).',
            current_revision=current.revision,
            report=report,
        )
    if class_conflicts and not body.unbind_conflicting_class_hotkeys:
        conflict = class_conflicts[0]
        raise api_error(
            409,
            'class_hotkey_conflict',
            f"'{conflict.combo}' is bound to class '{conflict.class_name}' in project "
            f'{conflict.project}.',
            current_revision=current.revision,
            class_conflicts=[
                {
                    'project': c.project,
                    'class_id': c.class_id,
                    'class_name': c.class_name,
                    'combo': c.combo,
                    'action_id': c.action_id,
                }
                for c in class_conflicts
            ],
        )

    unbound = (
        await _unbind_classes(class_conflicts) if body.unbind_conflicting_class_hotkeys else []
    )

    try:
        new_doc = await save_keymap_doc(
            opensearch,
            cfg.configs_index,
            overrides=merged,
            expected_revision=body.expected_revision,
        )
    except RevisionConflictError as exc:
        raise api_error(
            409,
            'revision_conflict',
            'The keymap changed since you loaded it.',
            current_revision=exc.current_revision,
        ) from exc

    await _publish_keymap_changed(new_doc)
    logger.info('keymap_reset', project=cfg.project_slug, revision=new_doc.revision)
    response = _build_response(new_doc, project=cfg.project_slug)
    return KeymapPutResponse(**response.model_dump(), unbound_class_hotkeys=unbound)
