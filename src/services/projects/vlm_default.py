"""A project calls no VLM until it activates one.

The activation doc is what says so. A project created now gets an explicit
``off`` (:func:`pin_new_project_vlm_off`, written while it is still
``building``). A project created by an earlier release has no doc and used to
inherit the deployment's ``env`` endpoint; :func:`pin_legacy_vlm` writes that
inheritance down once as an explicit activation (or ``off`` when the
deployment has no env endpoint), so its effective behaviour does not change and
a project with no doc is never read as "inherit". It only runs on ``active``
projects, so a project mid-create (always pinned off first) is never touched.
"""

from __future__ import annotations

from typing import Any

from src.core.logging import get_logger


logger = get_logger(__name__)


async def _pin(client: Any, endpoint_name: str | None) -> bool:
    """Write the activation doc when there is none (a quiet write: no
    ``config.changed`` event, no change to this process's snapshot, so a
    create's event order and a start-up repair stay unaffected). ``False``
    when the project already has one."""
    from src.services.config_store import get_config_store
    from src.services.config_store.index import activate, get_activation
    from src.services.labeling.vlm_endpoints import refresh_vlm_state

    await refresh_vlm_state(client)
    index = get_config_store().index
    if await get_activation(client, index, 'vlm') is not None:
        return False
    await activate(
        client,
        index,
        axis='vlm',
        name=endpoint_name,
        revision=None,
        expected_active=None,
        doc_fields={'acked_refs': {}, 'external_ack_at': None},
    )
    return True


async def pin_new_project_vlm_off(client: Any) -> None:
    """Write an explicit ``off`` for the bound, just-created project (a clone
    that copies the VLM axis writes its own activation afterwards)."""
    await _pin(client, None)


async def pin_legacy_vlm(client: Any) -> str | None:
    """For the bound project with no VLM activation doc: record the old
    inheritance explicitly. Returns what was written (``'env'`` / ``'off'``)
    or ``None`` when the project already has an activation."""
    from src.services.labeling.vlm_endpoints import env_builtin

    endpoint = env_builtin()
    name = None if endpoint is None else endpoint.name
    if not await _pin(client, name):
        return None
    return name or 'off'


__all__ = ['pin_legacy_vlm', 'pin_new_project_vlm_off']
