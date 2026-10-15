"""Refusal checks that run before any clone-settings write.

Split out of ``clone.py``: the source-readiness check and the full
validation of a clone request (status, axes, source, target emptiness).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.config.project_context import bind_project
from src.config.projects import CLONEABLE_AXES
from src.services.rejections import rejection


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
            raise rejection(422, 'combine_invalid', f"unknown clone axis '{axis}'")

    if from_slug == target_slug:
        raise rejection(
            422,
            'combine_invalid',
            f"'{target_slug}' cannot be cloned into itself",
            project=target_slug,
        )

    source = _require_found(await _resolve_existing(from_slug), from_slug)
    if source.status not in _CLONE_SOURCE_READY_STATUSES:
        raise rejection(
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
) -> tuple[ProjectRecord, list[str], dict[str, dict[str, Any] | None]]:
    """Every refusal a clone can hit, checked before anything is written:
    the source-shaped checks in :func:`_validate_clone_source`, plus
    ``classes`` into a target that already has items (409
    ``target_not_empty`` -- a clone is always a byte-identical starting
    point, never a merge). Returns the source record, resolved axes, and
    (N3 fix) each activation axis's REAL existing target doc (``None``
    when no doc exists at all) -- ``_clone_activations`` must use this as
    ``expected_active`` instead of assuming an empty target."""
    source, resolved_axes = await _validate_clone_source(
        client, target_slug=target_record.slug, from_slug=from_slug, axes=axes
    )

    if 'classes' in resolved_axes:
        with bind_project(target_record):
            from src.config.curation_accessors import items_index

            count_resp = await client.count(index=items_index())
        if (count_resp.get('count') or 0) > 0:
            raise rejection(
                409,
                'target_not_empty',
                f"'{target_record.slug}' already has items; classes cannot be cloned",
                project=target_record.slug,
            )

    if 'prompt_packs' in resolved_axes:
        # every STORED pack (not just the active one -- 'activations'
        # already covers that), so refuse up front if the target already
        # has any of its own, same "clone is a byte-identical starting
        # point, never a merge" rule as 'classes'.
        from src.services.config_store import get_config_store

        with bind_project(target_record):
            target_store = get_config_store()
            await target_store.ensure_fresh(client)
            if target_store.current.packs:
                raise rejection(
                    409,
                    'target_not_empty',
                    f"'{target_record.slug}' already has stored prompt packs; "
                    'prompt_packs cannot be cloned',
                    project=target_record.slug,
                )

    target_activations: dict[str, dict[str, Any] | None] = {}
    if 'activations' in resolved_axes:
        # M5: a clone is "every check before the first write" -- a
        # target that already has ITS OWN activation on either axis
        # would otherwise 409 (RevisionConflictError/ActiveConflictError,
        # from `save_config(expected_revision=None)` /
        # `activate(expected_active=None)` assuming an empty target)
        # only after `settings_defaults`/`classes` had already been
        # written. Refuse up front instead, same as `classes`.
        #
        # two more target-not-empty shapes
        # `existing and existing.get('name')` alone missed, both proven by
        # the reviewer to half-write before a 409 through
        # `clone_settings_into` (the real revision the old B2 fix was
        # scoped to, since a brand-new target can never hit either):
        #
        # (a) the target once DEACTIVATED this axis. That leaves a real
        #     stored activation doc shaped `{'name': None, 'revision':
        #     None}` -- truthy as a dict, so `existing and
        #     existing.get('name')` correctly treats it as "empty" here
        #     (no refusal), but `_clone_activations` then called
        #     `activate(..., expected_active=None)` unconditionally,
        #     which mismatches the *actual* stored previous value
        #     (`{'name': None, 'revision': None}` != Python `None`) and
        #     raises `ActiveConflictError` -- a false "already has an
        #     active X" 409 raised only after settings/keymap/pack writes
        #     had already landed. Fix: capture the REAL existing doc here
        #     (`None` only when no doc exists at all) and thread it
        #     through as `expected_active`, so the write-time check
        #     matches what is truly stored.
        # (b) the target has a saved, never-activated pack/profile
        #     sharing a NAME with the source's currently-active one on
        #     this axis. `existing.get('name')` is about the target's own
        #     active name, not a stored-but-inactive one, so this slipped
        #     through and `_clone_activations`'s `save_config(...,
        #     expected_revision=None)` treated the target's existing
        #     stored doc as "new", corrupting its revision history before
        #     the ensuing `ActiveConflictError`/`RevisionConflictError`.
        #     Fix: check the source's active NAME (per axis) against the
        #     target's config-store snapshot for either kind, up front.
        from opensearchpy.exceptions import NotFoundError

        from src.services.config_store import get_config_store
        from src.services.config_store.index import ConfigAxis, ConfigKind, get_activation

        activation_axes: tuple[ConfigAxis, ...] = ('prompt_pack', 'detection_profile')

        with bind_project(source, read_only=True):
            from src.config import get_curation_config as _get_src_cfg

            source_index = _get_src_cfg().configs_index
            source_active_names: dict[ConfigAxis, str | None] = {}
            # the revision goes
            # with the name below, both needed to re-resolve the
            # SOURCE's exact activated body for the target-context gate
            # check.
            source_active_refs: dict[ConfigAxis, tuple[str, int | None] | None] = {}
            for axis in activation_axes:
                src_activation = await get_activation(client, source_index, axis)
                src_name = (src_activation or {}).get('name')
                source_active_names[axis] = src_name
                source_active_refs[axis] = (
                    (src_name, (src_activation or {}).get('revision')) if src_name else None
                )

        with bind_project(target_record):
            from src.config import get_curation_config as _get_cfg

            target_index = _get_cfg().configs_index
            target_store = get_config_store()
            await target_store.ensure_fresh(client)
            for axis in activation_axes:
                # `get_activation` already maps a
                # real (or fake -- tests/projects/conftest.py's
                # ``FakeLifecycleOpenSearch.get`` now raises like the real
                # client) 404 to `None` itself; nothing here can still
                # raise `NotFoundError`/`KeyError`.
                existing = await get_activation(client, target_index, axis)
                if existing and existing.get('name'):
                    raise rejection(
                        409,
                        'target_not_empty',
                        f"'{target_record.slug}' already has an active {axis}; "
                        'activations cannot be cloned',
                        project=target_record.slug,
                    )
                # `activate()`'s own OCC check compares `expected_active`
                # against exactly `{'name':..., 'revision':...}` built
                # from the stored doc, not the raw `_source` (which also
                # carries `doc_type`/`axis`/`activated_at`/`previous`) --
                # normalize the same way here so this genuinely matches.
                target_activations[axis] = (
                    {'name': existing.get('name'), 'revision': existing.get('revision')}
                    if existing is not None
                    else None
                )

                source_name = source_active_names.get(axis)
                if source_name:
                    # `target_store.current`
                    # is a snapshot refreshed at most once per
                    # `ensure_fresh`'s 1s TTL -- a genuinely concurrent
                    # write from another worker landing in that window
                    # (e.g. another process just stored this exact name)
                    # is invisible here, so this check would pass and
                    # `_clone_activations`'s `save_config(...,
                    # expected_revision=None)` would then partially write
                    # before hitting `RevisionConflictError`. Read the
                    # specific doc straight from the client instead of
                    # trusting the cached snapshot.
                    from src.services.config_store.index import config_doc_id

                    kind: ConfigKind = 'prompt_pack' if axis == 'prompt_pack' else 'region_profile'
                    doc_id = config_doc_id(kind, source_name)
                    try:
                        await client.get(index=target_index, id=doc_id)
                        name_taken = True
                    except NotFoundError:
                        name_taken = False
                    if name_taken:
                        raise rejection(
                            409,
                            'target_not_empty',
                            f"'{target_record.slug}' already has a stored {axis} named "
                            f"'{source_name}'; activations cannot be cloned",
                            project=target_record.slug,
                        )

            # every check above is
            # SOURCE-shaped (does the target already have something in
            # the way). None of them ask whether the source's activated
            # pair is even VALID in the TARGET's own context -- see
            # ``clone_activation_gate.py`` (split out to stay under the
            # 700-LOC ratchet).
            from src.services.projects.clone_activation_gate import (
                check_activation_pair_in_target_context,
            )

            await check_activation_pair_in_target_context(
                client,
                source=source,
                target_record=target_record,
                source_active_refs=source_active_refs,
                activation_axes=activation_axes,
            )
    if 'open_vocab' in resolved_axes:
        from src.services.projects.clone_open_vocab import validate_open_vocab_clone

        target_activations['open_vocab'] = await validate_open_vocab_clone(
            client, target_record=target_record
        )
    if 'vlm_activation' in resolved_axes:
        from src.services.projects.clone_vlm import validate_vlm_activation_clone

        target_activations['vlm'] = await validate_vlm_activation_clone(
            client,
            source=source,
            target_record=target_record,
            with_activations='activations' in resolved_axes,
        )
    return source, resolved_axes, target_activations
