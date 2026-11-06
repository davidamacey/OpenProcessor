"""The ``vlm_activation`` axis of project ``clone_settings``
(``CLONEABLE_AXES`` includes ``'vlm_activation'``).

Copies which VLM endpoint the source project runs. The endpoint registry is
deployment-wide, so nothing else moves; what must NOT move is the
external-images acknowledgement: the target starts with no acknowledged
endpoints, and the activation goes through the same gate as every other
selection path (:func:`~src.services.config_store.vlm_activation.prepare_vlm_activation`,
mode ``clone``, bound to the TARGET). A source running an endpoint outside
this deployment therefore refuses the clone (422
``vlm_external_not_acknowledged``): retry without this axis, or activate the
endpoint in the new project yourself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.config.project_context import bind_project
from src.services.config_store.documents import ActivationRead, read_activation
from src.services.config_store.vlm_activation import (
    UNSET,
    PreparedVlmActivation,
    prepare_vlm_activation,
    write_vlm_activation,
)
from src.services.config_store.vlm_gate import ProjectVlmState
from src.services.rejections import rejection


if TYPE_CHECKING:
    from src.config.projects import ProjectRecord

_NOTHING_ACKED = ProjectVlmState(active_ref=None, active_ack_at=None, acked_refs=frozenset())


async def _pending_pack_profile(source: ProjectRecord) -> tuple[Any, Any]:
    """The source's activated pack / profile (decoded), which a clone that
    also copies ``activations`` makes the target's -- the VLM is paired with
    those, not with what the target runs before the clone."""
    from src.services.config_store.activation_gate import _decode_pack, _decode_profile

    out: dict[str, Any] = {'prompt_pack': UNSET, 'detection_profile': UNSET}
    for axis, decode in (('prompt_pack', _decode_pack), ('detection_profile', _decode_profile)):
        read = await read_activation(source.slug, axis)  # type: ignore[arg-type]
        if read is None or not read.name or read.revision is None or read.body is None:
            continue
        decoded = decode(read.name, read.body)
        if decoded is not None:
            out[axis] = decoded
    return out['prompt_pack'], out['detection_profile']


async def _prepare(
    *,
    source: ProjectRecord,
    target_record: ProjectRecord,
    with_activations: bool,
) -> tuple[ActivationRead | None, PreparedVlmActivation | None]:
    """``(source activation or None, gated target or None for "off")``.
    Raises the gate's rejection when the target may not run it."""
    doc = await read_activation(source.slug, 'vlm')
    if doc is None or not doc.name:
        return doc, None
    pack, profile = await _pending_pack_profile(source) if with_activations else (UNSET, UNSET)
    with bind_project(target_record):
        prepared = await prepare_vlm_activation(
            name=doc.name,
            revision=doc.revision,
            mode='clone',
            state=_NOTHING_ACKED,
            pack=pack,
            profile=profile,
        )
    return doc, prepared


async def validate_vlm_activation_clone(
    *,
    source: ProjectRecord,
    target_record: ProjectRecord,
    with_activations: bool,
) -> dict[str, Any] | None:
    """Every refusal, before the first write. Returns the ``expected_active``
    the write must present (the target's REAL activation, ``None`` when
    it has none)."""
    existing = await read_activation(target_record.slug, 'vlm')
    if existing is not None and existing.name:
        raise rejection(
            409,
            'target_not_empty',
            f"'{target_record.slug}' already has an active VLM; vlm_activation cannot be cloned",
            project=target_record.slug,
        )
    await _prepare(source=source, target_record=target_record, with_activations=with_activations)
    return existing.as_expected() if existing is not None else None


async def apply_vlm_activation_clone(
    *,
    source: ProjectRecord,
    target_record: ProjectRecord,
    expected_active: dict[str, Any] | None,
    with_activations: bool,
) -> None:
    """Re-run the gate against the target (a fresh look, not the validation's
    verdict) and write the activation."""
    doc, prepared = await _prepare(
        source=source, target_record=target_record, with_activations=with_activations
    )
    if doc is None:
        return
    with bind_project(target_record):
        await write_vlm_activation(
            prepared.endpoint if prepared else None,
            expected_active=expected_active,
            ack_now=False,
            reason='clone',
        )


__all__ = ['apply_vlm_activation_clone', 'validate_vlm_activation_clone']
