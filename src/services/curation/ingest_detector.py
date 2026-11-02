"""The ingest detectors a project runs: the deployment's ``OP_INGEST_*`` profiles,
or the project's active ``ingest_profile`` laid over them.

:func:`project_ingest_profiles` is the one place the active profile is resolved,
so ingest, the detector info, class seeding, a dataset import's propose mode,
the open-vocabulary validation, the model catalogs and the unload guard all see
the same detectors for a project. It reads the bound project's config-store
snapshot (synchronous, no new cache); async callers use
:func:`refreshed_ingest_profiles` to see another worker's activation first.

Fail closed: an unbound project raises, so does a store that never loaded or is
stale, and so does an activation whose pinned revision cannot be read or decoded.
Only "never activated" and "off", read from a loaded fresh store, fall back to
the environment. A stored profile's classes are never registry ids
(``assigns_class`` is forced off) and its ``name`` stays ``item`` / ``secondary``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pydantic import ValidationError

from src.config.detection_profile import DetectionProfile
from src.config.ingest_profiles import (
    INGEST_SECONDARY_ENV_PREFIX,
    ingest_primary_profile,
    ingest_secondary_profile,
)
from src.schemas.curation.ingest_profile import IngestDetectorSpec, IngestProfileBody
from src.services.config_store import get_config_store


if TYPE_CHECKING:
    from src.services.config_store.snapshot import StoredConfig


class IngestProfileUnresolvedError(RuntimeError):
    """The project's active ingest profile cannot be read; ingest refuses
    rather than falling back to the deployment default."""


@dataclass(frozen=True)
class IngestProfiles:
    primary: DetectionProfile
    secondary: DetectionProfile | None
    source: Literal['env', 'stored']
    ref: tuple[str, int] | None = None


def lay_spec(base: DetectionProfile, spec: IngestDetectorSpec) -> DetectionProfile:
    """``base`` with the stored detector spec laid over it; fields the spec leaves
    ``None`` keep ``base``'s value."""
    changes: dict[str, Any] = {
        'detector_model': spec.model,
        'detector_version': spec.version,
        'labels_path': spec.labels_path,
        'assigns_class': False,
    }
    for field in ('input_size', 'confidence_floor', 'batch_limit'):
        if getattr(spec, field) is not None:
            changes[field] = getattr(spec, field)
    return dataclasses.replace(base, **changes)


def _decode(pinned: StoredConfig) -> IngestProfileBody:
    try:
        return IngestProfileBody.model_validate(pinned.body)
    except ValidationError as exc:
        msg = (
            f'the active ingest profile {pinned.name!r} (revision {pinned.revision}) '
            f'is invalid: {exc}'
        )
        raise IngestProfileUnresolvedError(msg) from exc


def project_ingest_profiles() -> IngestProfiles:
    """The bound project's ingest detectors (raises when no project is bound).

    Refuses unless the snapshot is loaded and fresh: a store that never loaded or
    whose last refresh failed cannot say whether a profile is active, and the
    environment profile is only right when it can say none is."""
    current = get_config_store().current
    if not current.loaded_at:
        msg = 'the config store has not loaded; the active ingest profile is unknown'
        raise IngestProfileUnresolvedError(msg)
    if current.stale:
        msg = 'the config store could not be refreshed; the active ingest profile is unconfirmed'
        raise IngestProfileUnresolvedError(msg)
    ref = current.active_ingest_profile
    if not isinstance(ref, tuple):
        return IngestProfiles(
            primary=ingest_primary_profile(), secondary=ingest_secondary_profile(), source='env'
        )
    pinned = current.active_ingest_profile_body
    if pinned is None or ref[1] is None:
        msg = (
            f'the active ingest profile {ref[0]!r} could not be read from the config store; '
            'activate an ingest profile again or deactivate it'
        )
        raise IngestProfileUnresolvedError(msg)
    body = _decode(pinned)
    secondary_base = DetectionProfile.from_env(INGEST_SECONDARY_ENV_PREFIX, name='secondary')
    return IngestProfiles(
        primary=lay_spec(ingest_primary_profile(), body.primary),
        secondary=lay_spec(secondary_base, body.secondary) if body.secondary else None,
        source='stored',
        ref=(ref[0], ref[1]),
    )


async def refreshed_ingest_profiles(client: Any) -> IngestProfiles:
    """:func:`project_ingest_profiles` after refreshing the snapshot when it is
    over a second old, so an activation made through another worker is seen."""
    await get_config_store().ensure_fresh(client)
    return project_ingest_profiles()


__all__ = [
    'IngestProfileUnresolvedError',
    'IngestProfiles',
    'lay_spec',
    'project_ingest_profiles',
    'refreshed_ingest_profiles',
]
