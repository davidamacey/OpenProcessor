"""Reading and cloning the per-project activations (detection and region
profiles, VLM activation) for a clone-settings request.

Split out of ``clone.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.config.project_context import bind_project
from src.services.rejections import rejection


if TYPE_CHECKING:
    from src.config.projects import ProjectRecord


@dataclass(frozen=True)
class ActivatedConfig:
    name: str
    revision: int | None
    body: Any


async def read_activated_config(record: ProjectRecord, axis: Any) -> ActivatedConfig | None:
    """The config ``record`` has ACTIVE on ``axis``: its name, revision and the body that was
    activated (the immutable revision, not whatever the document holds now, which diverges once
    the source saves again without reactivating). ``None`` when the axis is off, unset, or its
    revision is gone."""
    from src.services.config_store.documents import read_activation

    read = await read_activation(record.slug, axis)
    # a registered config (no revision) is not copied: the target has it already
    if read is None or read.name is None or read.revision is None or read.body is None:
        return None
    return ActivatedConfig(name=read.name, revision=read.revision, body=read.body)


async def _clone_activations(
    *,
    target_record: ProjectRecord,
    source: ProjectRecord,
    written_packs: dict[str, Any] | None = None,
    target_activations: dict[str, dict[str, Any] | None] | None = None,
) -> None:
    """Copy the source's active
    ``prompt_pack``/``detection_profile`` -- the stored config body plus
    the activation itself -- into the target. A source axis that is
    ``off``, unset, or resolves to an env/file-registered profile (never
    written to the store) is skipped for that axis; the target simply
    keeps whatever it already had, which is empty for a brand-new
    project. Never raises on "nothing to clone" -- only on a genuine
    write failure."""

    from src.services.config_store.documents import ConfigAxis, ConfigKind, activate, save_config

    axis_kinds: tuple[tuple[ConfigAxis, ConfigKind], ...] = (
        ('prompt_pack', 'prompt_pack'),
        ('detection_profile', 'region_profile'),
    )
    for axis, kind in axis_kinds:
        activated = await read_activated_config(source, axis)
        if activated is None:
            continue
        name, body = activated.name, activated.body
        revision = activated.revision

        with bind_project(target_record):
            from src.services.config_store.documents import (
                ActiveConflictError,
                CloneSource,
                RevisionConflictError,
            )

            # the fix
            # re-read the target store's cache here to detect a sibling
            # `prompt_packs` axis write, but that cache has a 1s TTL
            # (`ensure_fresh`) -- a clone that finishes inside that window
            # (the common case) saw the stale, pre-write cache and 409'd
            # `target_not_empty` against its own just-written pack, after
            # settings/classes/keymap/packs had already landed. Take the
            # sibling axis's write as an explicit argument instead
            # (`written_packs`, only ever populated for kind='prompt_pack'
            # -- there is no bulk 'region_profiles' clone axis) so this
            # never needs to re-read what was just written.
            #
            # This must still hold even when the sibling axis wrote
            # first: the body actually made ACTIVE in the target is always
            # the source's ACTIVATED body (`body`, fetched above from the
            # immutable `<kind>:<name>@<rev>` copy) -- never the source's
            # merely-current one. When the `prompt_packs` axis already wrote
            # that exact same body (the common case: the source's current
            # doc is byte-identical to what's active), reuse its revision
            # instead of a redundant write. When the source has since
            # edited without reactivating (current diverges from
            # activated), the sibling axis's write only covers the
            # *current* doc -- a
            # second revision carrying the *activated* body is saved here
            # so the target ends up in the same current-vs-activated split
            # state as the source, and the target's active pack is never
            # promoted from an unvalidated draft.
            sibling_written = (written_packs or {}).get(name) if kind == 'prompt_pack' else None
            # M5 (defense in depth -- _validate_clone already refuses an
            # occupied target up front): a conflict here is still mapped
            # to a structured 409, never a bare 500, in case the target
            # changed between validation and this write.
            try:
                if sibling_written is not None and sibling_written.body == body:
                    revision = sibling_written.revision
                else:
                    saved, _ = await save_config(
                        kind=kind,
                        name=name,
                        body=body,
                        expected_revision=sibling_written.revision if sibling_written else None,
                        cloned_from=CloneSource(source.slug, kind, name, activated.revision),
                    )
                    revision = saved.revision
                # N3 fix: use the REAL existing target doc captured during
                # validation (`_validate_clone`) as `expected_active`,
                # not a bare `None` -- a target that once deactivated this
                # axis has a real stored doc shaped `{'name': None,
                # 'revision': None}`, which mismatches Python `None` and
                # raised a false 409 here, after every other axis had
                # already committed.
                await activate(
                    axis=axis,
                    name=name,
                    revision=revision,
                    expected_active=(target_activations or {}).get(axis),
                    reason='clone',
                )
            except RevisionConflictError as exc:
                raise rejection(
                    409,
                    'target_not_empty',
                    f"'{target_record.slug}' already has a stored {kind} named '{name}'",
                    project=target_record.slug,
                ) from exc
            except ActiveConflictError as exc:
                raise rejection(
                    409,
                    'target_not_empty',
                    f"'{target_record.slug}' already has an active {axis}",
                    project=target_record.slug,
                ) from exc
