"""The ``vlm_activation`` axis of project ``clone_settings`` (W9,
projects_plan.md §11 W9: ``CLONEABLE_AXES += {'vlm_activation'}``).

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
from src.routers.curation._config_common_models import api_error
from src.services.config_store.index import config_doc_id, get_activation
from src.services.config_store.vlm_activation import (
    UNSET,
    PreparedVlmActivation,
    prepare_vlm_activation,
    write_vlm_activation,
)
from src.services.config_store.vlm_gate import ProjectVlmState


if TYPE_CHECKING:
    from src.config.projects import ProjectRecord

_NOTHING_ACKED = ProjectVlmState(active_ref=None, active_ack_at=None, acked_refs=frozenset())


async def _source_activation(client: Any, source: ProjectRecord) -> dict[str, Any] | None:
    with bind_project(source, read_only=True):
        from src.config import get_curation_config

        return await get_activation(client, get_curation_config().configs_index, 'vlm')


async def _pending_pack_profile(client: Any, source: ProjectRecord) -> tuple[Any, Any]:
    """The source's activated pack / profile (decoded), which a clone that
    also copies ``activations`` makes the target's -- the VLM is paired with
    those, not with what the target runs before the clone."""
    from opensearchpy.exceptions import NotFoundError

    from src.services.config_store.activation_gate import _decode_pack, _decode_profile

    out: dict[str, Any] = {'prompt_pack': UNSET, 'detection_profile': UNSET}
    with bind_project(source, read_only=True):
        from src.config import get_curation_config

        index = get_curation_config().configs_index
        for axis, kind, decode in (
            ('prompt_pack', 'prompt_pack', _decode_pack),
            ('detection_profile', 'region_profile', _decode_profile),
        ):
            doc = await get_activation(client, index, axis)  # type: ignore[arg-type]
            name, revision = (doc or {}).get('name'), (doc or {}).get('revision')
            if not name or revision is None:
                continue
            try:
                stored = await client.get(
                    index=index,
                    id=config_doc_id(kind, name, revision),  # type: ignore[arg-type]
                )
            except NotFoundError:
                continue
            decoded = decode(name, (stored.get('_source') or {}).get('body') or {})
            if decoded is not None:
                out[axis] = decoded
    return out['prompt_pack'], out['detection_profile']


async def _prepare(
    client: Any,
    *,
    source: ProjectRecord,
    target_record: ProjectRecord,
    with_activations: bool,
) -> tuple[dict[str, Any] | None, PreparedVlmActivation | None]:
    """``(source activation doc or None, gated target or None for "off")``.
    Raises the gate's ``api_error`` when the target may not run it."""
    doc = await _source_activation(client, source)
    if doc is None or not doc.get('name'):
        return doc, None
    pack, profile = (
        await _pending_pack_profile(client, source) if with_activations else (UNSET, UNSET)
    )
    with bind_project(target_record):
        prepared = await prepare_vlm_activation(
            client,
            name=doc['name'],
            revision=doc.get('revision'),
            mode='clone',
            state=_NOTHING_ACKED,
            pack=pack,
            profile=profile,
        )
    return doc, prepared


async def validate_vlm_activation_clone(
    client: Any,
    *,
    source: ProjectRecord,
    target_record: ProjectRecord,
    with_activations: bool,
) -> dict[str, Any] | None:
    """Every refusal, before the first write. Returns the ``expected_active``
    the write must present (the target's REAL activation doc, ``None`` when
    it has none)."""
    with bind_project(target_record):
        from src.config import get_curation_config

        existing = await get_activation(client, get_curation_config().configs_index, 'vlm')
    if existing and existing.get('name'):
        raise api_error(
            409,
            'target_not_empty',
            f"'{target_record.slug}' already has an active VLM; vlm_activation cannot be cloned",
            project=target_record.slug,
        )
    await _prepare(
        client, source=source, target_record=target_record, with_activations=with_activations
    )
    return (
        {'name': existing.get('name'), 'revision': existing.get('revision')}
        if existing is not None
        else None
    )


async def apply_vlm_activation_clone(
    client: Any,
    *,
    source: ProjectRecord,
    target_record: ProjectRecord,
    expected_active: dict[str, Any] | None,
    with_activations: bool,
) -> None:
    """Re-run the gate against the target (a fresh look, not the validation's
    verdict) and write the activation."""
    doc, prepared = await _prepare(
        client, source=source, target_record=target_record, with_activations=with_activations
    )
    if doc is None:
        return
    with bind_project(target_record):
        await write_vlm_activation(
            client,
            prepared.endpoint if prepared else None,
            expected_active=expected_active,
            ack_now=False,
        )


__all__ = ['apply_vlm_activation_clone', 'validate_vlm_activation_clone']
