"""Target-context activation-gate check for project clone (R5-3 fix,
W3/W4 round-5 review, Major).

Split out of ``clone.py`` to stay under the repo's 700-LOC pre-commit
ratchet; ``clone.py``'s ``_validate_clone`` is the only caller.

Every OTHER check in ``_validate_clone`` is SOURCE-shaped (does the
target already have something in the way). None of them ask whether the
source's activated pair is even VALID in the TARGET's own context -- a
region profile can name a detector the source project owns privately
(``detector_model_not_shared``, non-bypassable), which only the gate run
bound to the TARGET (its class registry, Triton state, ``project_slug``)
catches. Copying that pair over anyway silently activates cross-project
model access the per-profile ``from_project`` clone route already
refuses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.config.project_context import bind_project
from src.routers.curation._config_common_models import api_error


if TYPE_CHECKING:
    from src.config.projects import ProjectRecord
    from src.services.config_store.index import ConfigAxis


async def check_activation_pair_in_target_context(
    client: Any,
    *,
    source: ProjectRecord,
    target_record: ProjectRecord,
    source_active_refs: dict[ConfigAxis, tuple[str, int | None] | None],
    activation_axes: tuple[ConfigAxis, ...],
) -> None:
    """Re-resolve each axis's SOURCE body, then run the shared gate bound
    to the TARGET, before any write. Raises ``api_error(422, ...)`` when
    the source's active pair does not pass ``for_activation`` validation
    in the target's own context.

    Scoped to ``detection_profile``: that is the axis with an actual
    target-specific validity condition today (``detector_model_not_
    shared``, via ``_check_detector``'s ``project_slug``). A standalone
    ``prompt_pack`` re-validation would ALSO run every pack-completeness/
    reply-key check ``for_activation`` implies -- correct in principle
    (pack vocabulary is target-class-registry-dependent too), but that is
    a materially bigger, unreviewed change with no reported gap behind
    it; scoping to the confirmed gap avoids rejecting pre-existing source
    packs that were never re-validated end-to-end. ``validate_profile``'s
    own embedded pack cross-check (``profile_validation.py``: ``if
    for_activation and active_pack is not None: ... validate_pack(...)``)
    still exercises the pack body via ``pending_sibling`` below, so a
    cross-axis failure (e.g. a multi-box-stripped pack paired with a
    multi-region profile) is still caught.
    """
    from src.services.config_store import get_config_store

    source_bodies: dict[ConfigAxis, dict[str, Any] | None] = {}
    for axis in activation_axes:
        ref = source_active_refs.get(axis)
        if ref is None:
            continue
        s_name, s_rev = ref
        src_record: Any = None
        with bind_project(source, read_only=True):
            src_store = get_config_store()
            await src_store.ensure_fresh(client)
            if axis == 'prompt_pack':
                from src.services.config_store.packs import (
                    build_record as build_src_pack_record,
                    get_revision_record as get_src_pack_revision,
                )

                src_record = build_src_pack_record(s_name, revision=s_rev)
                if src_record is None and s_rev is not None:
                    src_record = await get_src_pack_revision(client, s_name, s_rev)
            else:
                from src.services.config_store.profiles import (
                    build_record as build_src_profile_record,
                    get_revision_record as get_src_profile_revision,
                )

                src_record = build_src_profile_record(s_name, revision=s_rev)
                if src_record is None and s_rev is not None:
                    src_record = await get_src_profile_revision(client, s_name, s_rev)
        source_bodies[axis] = src_record.body if src_record is not None else None

    profile_ref = source_active_refs.get('detection_profile')
    profile_body = source_bodies.get('detection_profile')
    if profile_ref is None or profile_body is None:
        return

    from fastapi import HTTPException

    from src.services.config_store.activation_gate import run_activation_gate
    from src.services.labeling.vlm_prompts import PromptPack

    p_name, p_rev = profile_ref
    pending_kwargs: dict[str, Any] = {}
    pack_ref = source_active_refs.get('prompt_pack')
    pack_body = source_bodies.get('prompt_pack')
    if pack_ref is not None and pack_body is not None:
        pending_kwargs['pending_sibling'] = PromptPack.from_dict({**pack_body, 'name': pack_ref[0]})
    try:
        await run_activation_gate(
            'detection_profile', p_name, p_rev, client=client, body=profile_body, **pending_kwargs
        )
    except HTTPException as exc:
        raise api_error(
            422,
            'validation_failed',
            f"'{target_record.slug}' cannot clone active detection_profile "
            f"'{p_name}' from '{source.slug}': it does not pass activation "
            "validation in the target project's own context (e.g. a detector "
            'the target does not own or share)',
            project=target_record.slug,
        ) from exc


__all__ = ['check_activation_pair_in_target_context']
