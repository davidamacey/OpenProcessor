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
from src.services.projects.clone_activation import _clone_activations
from src.services.projects.clone_validate import _validate_clone
from src.utils.timeutil import now_iso


logger = get_logger(__name__)


if TYPE_CHECKING:
    from src.config.projects import ProjectRecord


async def _apply_clone(
    client: Any,
    *,
    target_record: ProjectRecord,
    source: ProjectRecord,
    axes: list[str],
    target_activations: dict[str, dict[str, Any] | None] | None,
) -> list[dict[str, Any]]:
    """Copy the validated axes. Reads the source under a read-only bind so
    the guard rejects any accidental write to it. Returns the ``keymap``
    axis's dropped-action conflicts (``[]`` for every other axis/outcome)
    -- a structured report, never a silent unbind.

    ``target_activations`` has no default
    -- every caller must say explicitly whether the target's per-axis
    activation state is empty (``None``) or the real
    ``_validate_clone``-captured doc (N3a). A silent ``None`` default
    would let a future caller reintroduce the exact "assume the target is
    empty" bug N3a fixed.
    """
    conflicts: list[dict[str, Any]] = []
    if 'settings_defaults' in axes:
        from src.services.projects.clone_settings import clone_settings_document

        await clone_settings_document(client, source=source, target_record=target_record)

    if 'classes' in axes:
        src_path = source.resources.class_registry_path
        dst_path = target_record.resources.class_registry_path
        if src_path.exists():
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)
        with bind_project(target_record):
            from src.clients.curation_opensearch.registry import ClassRegistry

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
        from src.services.curation.keymap import (
            get_keymap_doc,
            publish_keymap_changed,
            save_keymap_doc,
        )
        from src.services.curation.keymap_validator import validate_keymap

        with bind_project(source, read_only=True):
            src_cfg = get_curation_config()
            source_keymap = await get_keymap_doc(client, src_cfg.configs_index)
        with bind_project(target_record):
            target_cfg = get_curation_config()
            target_keymap = await get_keymap_doc(client, target_cfg.configs_index)
            from src.clients.curation_opensearch.registry import get_class_registry

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
                # Tabs already open on the target should refresh.
                await publish_keymap_changed(new_target_doc, target_cfg.configs_index, client)

    written_packs: dict[str, Any] = {}
    if 'prompt_packs' in axes:
        from src.services.projects.clone_stored import copy_stored_configs

        written_packs = await copy_stored_configs(
            client,
            target_record=target_record,
            source=source,
            kind='prompt_pack',
            stored_field='packs',
        )

    if 'activations' in axes:
        await _clone_activations(
            client,
            target_record=target_record,
            source=source,
            written_packs=written_packs,
            target_activations=target_activations or {},
        )

    if 'open_vocab' in axes:
        from src.services.projects.clone_open_vocab import apply_open_vocab_clone

        await apply_open_vocab_clone(
            client,
            source=source,
            target_record=target_record,
            expected_active=(target_activations or {}).get('open_vocab'),
        )

    if 'vlm_activation' in axes:
        from src.services.projects.clone_vlm import apply_vlm_activation_clone

        await apply_vlm_activation_clone(
            client,
            source=source,
            target_record=target_record,
            expected_active=(target_activations or {}).get('vlm'),
            with_activations='activations' in axes,
        )

    return conflicts


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
    source, resolved_axes, target_activations = await _validate_clone(
        client, target_record=target_record, from_slug=from_slug, axes=axes
    )
    return await _apply_clone(
        client,
        target_record=target_record,
        source=source,
        axes=resolved_axes,
        target_activations=target_activations,
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
        _require_revision,
        _require_transition,
        write_record,
    )
    from src.services.projects.registry import get_project_registry

    record, seq, term = await _get_mutable_record(client, slug)
    _require_transition(record, 'clone settings into', frozenset({'active'}))
    _require_revision(record, expected_revision)
    source, resolved_axes, target_activations = await _validate_clone(
        client, target_record=record, from_slug=from_slug, axes=axes
    )
    conflicts = await _apply_clone(
        client,
        target_record=record,
        source=source,
        axes=resolved_axes,
        target_activations=target_activations,
    )
    updated = replace(record, revision=record.revision + 1, updated_at=now_iso())
    await write_record(client, updated, if_seq_no=seq, if_primary_term=term)
    await get_project_registry().ensure_fresh()
    return updated, conflicts


__all__ = ['clone_settings', 'clone_settings_into']
