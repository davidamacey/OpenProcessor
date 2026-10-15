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


async def read_activated_config(
    client: Any, record: ProjectRecord, axis: Any, kind: Any
) -> ActivatedConfig | None:
    """The config ``record`` has ACTIVE on ``axis``: its name, revision and
    the body that was activated (the immutable ``<kind>:<name>@<rev>`` copy,
    not whatever ``<kind>:<name>`` holds now, which diverges once the source
    saves again without reactivating). ``None`` when the axis is off, unset,
    an env/file id that was never stored (no revision copy), or its revision
    copy is gone. A malformed stored doc raises."""
    from opensearchpy.exceptions import NotFoundError

    from src.config import get_curation_config
    from src.services.config_store.index import config_doc_id, get_activation

    with bind_project(record, read_only=True):
        index = get_curation_config().configs_index
        activation = await get_activation(client, index, axis)
        if not activation or not activation.get('name'):
            return None
        name = activation['name']
        revision = activation.get('revision')
        try:
            stored = await client.get(index=index, id=config_doc_id(kind, name, revision))
        except NotFoundError:
            return None
    body = (stored.get('_source') or {}).get('body')
    if body is None:
        return None
    return ActivatedConfig(name=name, revision=revision, body=body)


async def _clone_activations(
    client: Any,
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

    from src.services.config_store.index import ConfigAxis, ConfigKind, activate, save_config

    axis_kinds: tuple[tuple[ConfigAxis, ConfigKind], ...] = (
        ('prompt_pack', 'prompt_pack'),
        ('detection_profile', 'region_profile'),
    )
    for axis, kind in axis_kinds:
        activated = await read_activated_config(client, source, axis, kind)
        if activated is None:
            continue
        name, body = activated.name, activated.body
        revision = activated.revision

        with bind_project(target_record):
            from src.config import get_curation_config as _get_cfg
            from src.services.config_store.index import ActiveConflictError, RevisionConflictError

            target_index = _get_cfg().configs_index
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
                    doc = await save_config(
                        client,
                        target_index,
                        kind=kind,
                        name=name,
                        body=body,
                        expected_revision=sibling_written.revision if sibling_written else None,
                        cloned_from=source.slug,
                    )
                    revision = doc['revision']
                # N3 fix: use the REAL existing target doc captured during
                # validation (`_validate_clone`) as `expected_active`,
                # not a bare `None` -- a target that once deactivated this
                # axis has a real stored doc shaped `{'name': None,
                # 'revision': None}`, which mismatches Python `None` and
                # raised a false 409 here, after every other axis had
                # already committed.
                await activate(
                    client,
                    target_index,
                    axis=axis,
                    name=name,
                    revision=revision,
                    expected_active=(target_activations or {}).get(axis),
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
