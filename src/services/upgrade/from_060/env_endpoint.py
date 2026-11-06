"""The ``env`` VLM endpoint as a stored deployment-wide document.

In 0.6.0 a project could activate the built-in endpoint that ``OP_VLM_URL`` / ``OP_VLM_MODEL``
describe; it had no document and no revision. 0.7.0 gives an activation a real document to point
at, so activating the built-in stores its body as a system revision of the reserved ``env``
endpoint. The upgrade does the same once, with the body the running environment gives (the same
:func:`~src.services.labeling.vlm_endpoints.env_endpoint` the API serves), and keeps the built-in's
last probe on that revision while it still describes that body.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.services.config_store.vlm_endpoints import ENV_DESCRIPTION
from src.services.labeling.vlm_endpoints import env_endpoint
from src.services.upgrade.from_060.model import (
    ENV_NAME,
    ENV_REVISION,
    ConfigDocSrc,
    ConfigRevisionSrc,
    ProbeSrc,
    UnresolvedBuiltinError,
    VlmEndpointSrc,
)


if TYPE_CHECKING:
    import datetime as dt

    from src.services.upgrade.from_060.model import ActivationSrc, GlobalData, ProjectData


def _uses_env(activations: tuple[ActivationSrc, ...]) -> bool:
    return any(e.axis == 'vlm' and ENV_NAME in (e.to_name, e.from_name) for e in activations)


def endpoints_to_store(
    global_data: GlobalData, projects: list[ProjectData], *, now: dt.datetime
) -> list[VlmEndpointSrc]:
    """Every deployment-wide endpoint document: the stored ones, plus the ``env`` built-in when a
    project activated it and the environment still names a VLM."""
    endpoints = list(global_data.endpoints)
    if any(_uses_env(pd.activations) for pd in projects):
        endpoints.append(env_endpoint_source(global_data.env_probe, now=now))
    return endpoints


def env_endpoint_source(probe: ProbeSrc | None, *, now: dt.datetime) -> VlmEndpointSrc:
    """The endpoint document to store. Raises :class:`UnresolvedBuiltinError` when the environment
    names no VLM: a project's choice of the ``env`` endpoint is never dropped."""
    endpoint = env_endpoint(dict(probe.result) if probe else None)
    if endpoint is None:
        msg = (
            'a project activated the built-in VLM endpoint "env", but OP_VLM_URL is not set where '
            'the upgrade runs; set OP_VLM_URL / OP_VLM_MODEL as they were on 0.6.0 and run it again'
        )
        raise UnresolvedBuiltinError(msg)
    doc = ConfigDocSrc(
        project_slug=None,
        kind='vlm_endpoint',
        name=ENV_NAME,
        description=ENV_DESCRIPTION,
        revisions=(
            ConfigRevisionSrc(
                revision=ENV_REVISION,
                body=endpoint.body.model_dump(),
                created_at=now,
                cloned_from=None,
            ),
        ),
        created_at=now,
        updated_at=now,
    )
    # a probe of a body the environment no longer describes would never be shown: it is not kept
    kept = (probe,) if probe is not None and endpoint.last_probe is not None else ()
    return VlmEndpointSrc(doc=doc, probes=kept)
