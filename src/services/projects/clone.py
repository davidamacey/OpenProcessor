"""``clone_settings`` (§4): copy one project's settings/classes into
another, either during creation or into an existing ``active`` project.

Split out of ``lifecycle.py`` to stay under the repo's 700-LOC
pre-commit ratchet; ``lifecycle.py`` re-exports :func:`clone_settings`
and :func:`clone_settings_into` so every existing caller
(``lifecycle.clone_settings(...)``) keeps working unchanged.
"""

from __future__ import annotations

import shutil
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from src.config.project_context import bind_project
from src.core.logging import get_logger
from src.routers.curation._config_common_models import api_error
from src.routers.curation._project_models import CLONEABLE_AXES


logger = get_logger(__name__)


if TYPE_CHECKING:
    from src.config.projects import ProjectRecord


_CLONE_SOURCE_READY_STATUSES = frozenset({'active', 'archived'})


async def _validate_clone_source(
    client: Any,  # noqa: ARG001 - kept for signature symmetry with _validate_clone
    *,
    target_slug: str,
    from_slug: str,
    axes: list[str] | None,
) -> tuple[ProjectRecord, list[str]]:
    """Every clone refusal that does not depend on the target already
    existing, checked before the *first* write on either caller's path
    (M7): unknown axis (422 ``combine_invalid``), a clone into itself
    (422 ``combine_invalid`` -- was a 500 ``SameFileError``, m9), and a
    source that does not exist or is not ``active``/``archived`` (m9 --
    ``building``/``failed``/``deleting`` sources are refused, 409
    ``clone_source_not_ready``). Returns the source record and resolved
    axes; callers still run their own target-shaped checks (target
    emptiness) afterwards."""
    from src.services.projects.lifecycle import _require_found, _resolve_existing

    resolved_axes = axes if axes else list(CLONEABLE_AXES)
    for axis in resolved_axes:
        if axis not in CLONEABLE_AXES:
            raise api_error(422, 'combine_invalid', f"unknown clone axis '{axis}'")

    if from_slug == target_slug:
        raise api_error(
            422,
            'combine_invalid',
            f"'{target_slug}' cannot be cloned into itself",
            project=target_slug,
        )

    source = _require_found(await _resolve_existing(from_slug), from_slug)
    if source.status not in _CLONE_SOURCE_READY_STATUSES:
        raise api_error(
            409,
            'clone_source_not_ready',
            f"'{from_slug}' is {source.status}; only an active or archived project can be cloned",
            project=from_slug,
            project_status=source.status,
        )
    return source, resolved_axes


async def _validate_clone(
    client: Any,
    *,
    target_record: ProjectRecord,
    from_slug: str,
    axes: list[str] | None,
) -> tuple[ProjectRecord, list[str]]:
    """Every refusal a clone can hit, checked before anything is written:
    the source-shaped checks in :func:`_validate_clone_source`, plus
    ``classes`` into a target that already has items (409
    ``target_not_empty`` -- a clone is always a byte-identical starting
    point, never a merge). Returns the source record and resolved axes."""
    source, resolved_axes = await _validate_clone_source(
        client, target_slug=target_record.slug, from_slug=from_slug, axes=axes
    )

    if 'classes' in resolved_axes:
        with bind_project(target_record):
            from src.config.curation import items_index

            count_resp = await client.count(index=items_index())
        if (count_resp.get('count') or 0) > 0:
            raise api_error(
                409,
                'target_not_empty',
                f"'{target_record.slug}' already has items; classes cannot be cloned",
                project=target_record.slug,
            )

    if 'activations' in resolved_axes:
        # M5: a clone is "every check before the first write" -- a
        # target that already has ITS OWN activation on either axis
        # would otherwise 409 (RevisionConflictError/ActiveConflictError,
        # from `save_config(expected_revision=None)` /
        # `activate(expected_active=None)` assuming an empty target)
        # only after `settings_defaults`/`classes` had already been
        # written. Refuse up front instead, same as `classes`.
        from src.services.config_store.index import ConfigAxis, get_activation

        with bind_project(target_record):
            from src.config import get_curation_config as _get_cfg

            target_index = _get_cfg().configs_index
            activation_axes: tuple[ConfigAxis, ...] = ('prompt_pack', 'detection_profile')
            for axis in activation_axes:
                # Minor 3 (W2 review): `get_activation` already maps a
                # real (or fake -- tests/projects/conftest.py's
                # ``FakeLifecycleOpenSearch.get`` now raises like the real
                # client) 404 to `None` itself; nothing here can still
                # raise `NotFoundError`/`KeyError`.
                existing = await get_activation(client, target_index, axis)
                if existing and existing.get('name'):
                    raise api_error(
                        409,
                        'target_not_empty',
                        f"'{target_record.slug}' already has an active {axis}; "
                        'activations cannot be cloned',
                        project=target_record.slug,
                    )
    return source, resolved_axes


async def _apply_clone(
    client: Any, *, target_record: ProjectRecord, source: ProjectRecord, axes: list[str]
) -> list[dict[str, Any]]:
    """Copy the validated axes. Reads the source under a read-only bind so
    the guard rejects any accidental write to it. Returns the ``keymap``
    axis's dropped-action conflicts (``[]`` for every other axis/outcome)
    -- a structured report, never a silent unbind."""
    conflicts: list[dict[str, Any]] = []
    from src.clients.curation_opensearch import get_curation_settings, update_curation_settings

    if 'settings_defaults' in axes:
        with bind_project(source, read_only=True):
            source_settings = await get_curation_settings(client)
        with bind_project(target_record):
            await update_curation_settings(client, dict(source_settings.get('defaults', {})))

    if 'classes' in axes:
        src_path = source.resources.class_registry_path
        dst_path = target_record.resources.class_registry_path
        if src_path.exists():
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)
        with bind_project(target_record):
            from src.clients.curation_opensearch import ClassRegistry

            registry = ClassRegistry(dst_path)
            await registry.sync_to_opensearch(client)
    else:
        # No class registry was copied -- the target's registry is empty
        # (or byte-identical to whatever it already had), so the active
        # region profile's class (if any) would otherwise never exist in
        # it. Idempotent; a no-op when no region profile names one.
        from src.services.curation.region_class import ensure_region_class

        with bind_project(target_record):
            ensure_region_class()

    if 'keymap' in axes:
        from src.config import get_curation_config
        from src.services.curation.keymap import get_keymap_doc, save_keymap_doc
        from src.services.curation.keymap_validator import validate_keymap

        with bind_project(source, read_only=True):
            src_cfg = get_curation_config()
            source_keymap = await get_keymap_doc(client, src_cfg.configs_index)
        with bind_project(target_record):
            target_cfg = get_curation_config()
            target_keymap = await get_keymap_doc(client, target_cfg.configs_index)
            from src.routers.curation import get_class_registry

            target_classes = [
                {
                    'class_id': c.class_id,
                    'class_name': c.class_name,
                    'hotkey_letter': c.hotkey_letter,
                    'deprecated': c.deprecated,
                }
                for c in get_class_registry().load().classes
            ]
            report, class_conflicts, _resolved, _issues = validate_keymap(
                source_keymap.overrides,
                project=target_record.slug,
                classes=target_classes,
                previous_overrides=target_keymap.overrides,
            )
            # B2: all-or-nothing. Dropping the conflicting actions and
            # writing the rest used to leave those actions on their
            # *defaults*, which can themselves collide with a kept
            # override or with the same class -- an invalid keymap could
            # get written with no error and no GET issue reporting it.
            # A clash is a report, never a silent partial write (CW-K §0
            # clause 1): on any error or class conflict, the target's
            # keymap is left exactly as it was.
            conflicts = [
                {
                    'action_id': c.action_id,
                    'combo': c.combo,
                    'class_id': c.class_id,
                    'class_name': c.class_name,
                }
                for c in class_conflicts
            ]
            if class_conflicts or not report.ok:
                logger.warning(
                    'keymap_clone_conflicts_left_unchanged',
                    target=target_record.slug,
                    from_slug=source.slug,
                    conflicts=conflicts,
                    errors=[i.code for i in report.errors],
                )
            else:
                new_target_doc = await save_keymap_doc(
                    client,
                    target_cfg.configs_index,
                    overrides=source_keymap.overrides,
                    expected_revision=target_keymap.revision,
                )
                # Minor: tabs already open on the target should refresh.
                from src.services.curation.event_hub import get_event_hub

                get_event_hub().publish(
                    {
                        'type': 'config.changed',
                        'topic': 'config',
                        'axis': 'keymap',
                        'name': None,
                        'keymap_revision': new_target_doc.revision,
                    }
                )

    if 'activations' in axes:
        await _clone_activations(client, target_record=target_record, source=source)

    return conflicts


async def _clone_activations(
    client: Any, *, target_record: ProjectRecord, source: ProjectRecord
) -> None:
    """Glue G1 (projects_plan.md §11 W2): copy the source's active
    ``prompt_pack``/``detection_profile`` -- the stored config body plus
    the activation itself -- into the target. A source axis that is
    ``off``, unset, or resolves to an env/file-registered profile (never
    written to the store) is skipped for that axis; the target simply
    keeps whatever it already had, which is empty for a brand-new
    project. Never raises on "nothing to clone" -- only on a genuine
    write failure."""
    from opensearchpy.exceptions import NotFoundError

    from src.services.config_store.index import (
        ConfigAxis,
        ConfigKind,
        activate,
        config_doc_id,
        get_activation,
        save_config,
    )

    axis_kinds: tuple[tuple[ConfigAxis, ConfigKind], ...] = (
        ('prompt_pack', 'prompt_pack'),
        ('detection_profile', 'region_profile'),
    )
    for axis, kind in axis_kinds:
        with bind_project(source, read_only=True):
            from src.config import get_curation_config as _get_cfg

            source_index = _get_cfg().configs_index
            # Minor 3 (W2 review): `get_activation` already maps a real
            # (or fake) 404 to `None` itself -- nothing here can still
            # raise `NotFoundError`/`KeyError` to catch.
            activation = await get_activation(client, source_index, axis)
            if not activation or not activation.get('name'):
                continue
            name = activation['name']
            # Minor 2 (W2 review): copy the body that was actually
            # ACTIVATED (the immutable `<kind>:<name>@<rev>` revision
            # copy), not whatever `<kind>:<name>` (current) happens to
            # hold now -- the two diverge once the source saves again
            # without reactivating. `revision=None` (an env/file id,
            # never written to the store) has no revision copy to read;
            # the lookup below 404s and this axis is skipped, same as
            # "nothing to clone".
            revision = activation.get('revision')
            # Minor 3 (W2 review): a genuine 404 here is the only expected
            # failure (the activated revision copy no longer exists, e.g.
            # a never-stored env/file id); a malformed real doc should
            # raise, not be silently skipped, so `KeyError` is not caught.
            try:
                stored = await client.get(
                    index=source_index, id=config_doc_id(kind, name, revision)
                )
            except NotFoundError:
                continue
            body = (stored.get('_source') or {}).get('body')
            if body is None:
                continue

        with bind_project(target_record):
            from src.config import get_curation_config as _get_cfg
            from src.services.config_store.index import ActiveConflictError, RevisionConflictError

            target_index = _get_cfg().configs_index
            # M5 (defense in depth -- _validate_clone already refuses an
            # occupied target up front): a conflict here is still mapped
            # to a structured 409, never a bare 500, in case the target
            # changed between validation and this write.
            try:
                doc = await save_config(
                    client,
                    target_index,
                    kind=kind,
                    name=name,
                    body=body,
                    expected_revision=None,
                    cloned_from=source.slug,
                )
                await activate(
                    client,
                    target_index,
                    axis=axis,
                    name=name,
                    revision=doc['revision'],
                    expected_active=None,
                )
            except RevisionConflictError as exc:
                raise api_error(
                    409,
                    'target_not_empty',
                    f"'{target_record.slug}' already has a stored {kind} named '{name}'",
                    project=target_record.slug,
                ) from exc
            except ActiveConflictError as exc:
                raise api_error(
                    409,
                    'target_not_empty',
                    f"'{target_record.slug}' already has an active {axis}",
                    project=target_record.slug,
                ) from exc


async def clone_settings(
    client: Any,
    *,
    target_record: ProjectRecord,
    from_slug: str,
    axes: list[str] | None,
) -> list[dict[str, Any]]:
    """§4 ``clone_settings`` into a project being created: validate every
    refusal first, then copy. Returns the ``keymap`` axis's dropped-action
    conflicts (a structured report, never a silent unbind) -- ``[]`` for
    every other axis/outcome. ``create_project`` (M7) logs these today
    rather than threading them through its own return shape, which every
    other project-lifecycle test call site also unpacks."""
    source, resolved_axes = await _validate_clone(
        client, target_record=target_record, from_slug=from_slug, axes=axes
    )
    return await _apply_clone(
        client, target_record=target_record, source=source, axes=resolved_axes
    )


async def clone_settings_into(
    client: Any,
    *,
    slug: str,
    from_slug: str,
    axes: list[str] | None,
    expected_revision: int,
) -> tuple[ProjectRecord, list[dict[str, Any]]]:
    """§4 ``POST /projects/{project}/clone_settings`` into an existing
    ``active`` project. Every check (status, revision, axes, source,
    target emptiness) runs before anything is written, and the revision
    is bumped only after the copy succeeded -- a refused clone never
    changes the target's revision."""
    from src.services.projects.lifecycle import (
        _get_mutable_record,
        _now,
        _require_revision,
        _require_transition,
        write_record,
    )
    from src.services.projects.registry import get_project_registry

    record, seq, term = await _get_mutable_record(client, slug)
    _require_transition(record, 'clone settings into', frozenset({'active'}))
    _require_revision(record, expected_revision)
    source, resolved_axes = await _validate_clone(
        client, target_record=record, from_slug=from_slug, axes=axes
    )
    conflicts = await _apply_clone(client, target_record=record, source=source, axes=resolved_axes)
    updated = replace(record, revision=record.revision + 1, updated_at=_now())
    await write_record(client, updated, if_seq_no=seq, if_primary_term=term)
    await get_project_registry().ensure_fresh()
    return updated, conflicts


__all__ = ['clone_settings', 'clone_settings_into']
