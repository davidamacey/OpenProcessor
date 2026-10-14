"""The labeler one VLM run uses: its prompt pack, the endpoint, and the provenance stamp.

Shared by the VLM routes, the auto-label run and the health probe.
"""

from __future__ import annotations

from typing import Any


#: The one message meaning "this project's VLM is off"; :func:`get_vlm_labeler`
#: raises it and the routes map it to 409 ``vlm_not_configured``.
NO_VLM_MESSAGE = 'no VLM endpoint is active for this project'


def resolve_pack(pack_name: str | None, revision: int | None = None) -> Any:
    """The prompt pack a labeler is built with: the config store's *active*
    pack when ``pack_name`` is ``None`` (the activated pack, else the
    ``OP_PROMPT_PACK_PATH`` pack or the built-in generic pack), else the
    named pack (``revision`` pins an exact saved revision, ``name@rev``,
    §3.7). Unknown name -> ``ValueError``."""
    from src.services.labeling.vlm_prompts import active_prompt_pack, get_prompt_pack

    pack = (
        active_prompt_pack() if pack_name is None else get_prompt_pack(pack_name, revision=revision)
    )
    if pack is None:
        msg = f'unknown prompt pack {pack_name!r}'
        raise ValueError(msg)
    return pack


def get_vlm_labeler(
    pack_name: str | None = None, revision: int | None = None, *, endpoint: Any = None
) -> Any:
    """The labeler for ``(endpoint, pack)`` -- a thin wrapper over the one
    factory (:func:`~src.services.labeling.vlm_factory.labeler_for`).

    ``endpoint=None`` uses the bound project's ACTIVE endpoint from the
    in-process snapshots (callers on an async path refreshed them first,
    ``refresh_vlm_state``); routes that resolved a per-run ``?vlm=`` pass it
    explicitly. Raises ``VlmEndpointUnavailableError`` when there is none.
    """
    from src.services.labeling.vlm_endpoints import VlmEndpointUnavailableError, active_vlm_endpoint
    from src.services.labeling.vlm_factory import labeler_for

    pack = resolve_pack(pack_name, revision)
    resolved = endpoint if endpoint is not None else active_vlm_endpoint()
    if resolved is None:
        raise VlmEndpointUnavailableError(NO_VLM_MESSAGE)
    return labeler_for(resolved, pack)


def vlm_stamp(labeler: Any) -> dict[str, str]:
    """Provenance of a VLM-derived write: which endpoint/model answered."""
    return {
        'vlm_endpoint': labeler.identity.endpoint_ref,
        'vlm_model': labeler.identity.model,
    }
