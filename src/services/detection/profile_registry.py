"""In-memory registry of region :class:`~src.config.DetectionProfile`
instances, and resolution of the deployment's *active* region profile.

**Neutral by default.** An unconfigured deployment registers no profile:
:func:`get_active_region_profile` returns ``None``, ``GET /curation/methods``
advertises an empty ``detection_profile`` axis, and region detection (the
detection worker's cascade) stays off. Consumers must treat ``None`` as
"no region detection configured" and degrade cleanly.

A deployment configures its region profile purely through the environment:

- ``OP_REGION_PROFILE=<name>`` selects a profile by name — one already
  registered by startup code, or a built-in reference profile from
  :mod:`src.services.detection.reference_profiles` (e.g. ``license_plate``).
  An unknown name raises ``ValueError``.
- ``OP_REGION_DETECTION_<FIELD>`` overrides individual fields on top of the
  selected profile (e.g. ``OP_REGION_DETECTION_SAM_TEXT_PROMPT``,
  ``OP_REGION_DETECTION_SECONDARY_SHAPE_GROUPS``). With no
  ``OP_REGION_PROFILE`` these build a profile from the dataclass defaults
  (named by ``OP_REGION_DETECTION_NAME``, else ``region``).

The resolved profile is :func:`register_profile`'d as the default, so it
is exactly what ``GET /methods`` advertises. Note this is a separate
prefix from ``OP_DETECTION_*``, which configures the *ingest item
detector* (``routers/curation/ingest.py``), a different model with
different settings.

Further profiles (a deployment that detects more than one region type)
are added by startup code via :func:`register_profile`; every registered
profile is advertised and selectable by name.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from src.config import DetectionProfile

REGION_DETECTION_ENV_PREFIX = 'OP_REGION_DETECTION_'

_REGISTRY: dict[str, DetectionProfile] = {}
_DEFAULT_NAME: str | None = None
_ENV_RESOLVED = False


def register_profile(profile: DetectionProfile, *, default: bool = False) -> None:
    """Register (or replace) a profile under its own ``name``.

    The first profile ever registered becomes the default automatically;
    pass ``default=True`` to make a later registration the default instead
    (e.g. an operator explicitly re-pointing the default region type).
    """
    global _DEFAULT_NAME  # noqa: PLW0603 - module-level registry, mirrors CurationConfig's singleton pattern
    _REGISTRY[profile.name] = profile
    if default or _DEFAULT_NAME is None:
        _DEFAULT_NAME = profile.name


def region_profile_from_env() -> DetectionProfile | None:
    """Resolve the region profile the environment asks for (see module
    docstring), without registering it. ``None`` when nothing is configured.
    """
    from src.config import DetectionProfile
    from src.services.detection.reference_profiles import REFERENCE_PROFILES

    name = os.environ.get('OP_REGION_PROFILE', '').strip()
    has_overrides = DetectionProfile.env_overrides_present(REGION_DETECTION_ENV_PREFIX)
    if not name and not has_overrides:
        return None
    base: DetectionProfile | None = None
    if name:
        base = _REGISTRY.get(name) or REFERENCE_PROFILES.get(name)
        if base is None:
            known = sorted(set(_REGISTRY) | set(REFERENCE_PROFILES))
            msg = f'OP_REGION_PROFILE={name!r} is not a registered or built-in profile; known: {known}'
            raise ValueError(msg)
    return DetectionProfile.from_env(REGION_DETECTION_ENV_PREFIX, name='region', base=base)


def ensure_env_region_profile() -> None:
    """Resolve the env-configured region profile once per process and
    register it as the default. Idempotent; a no-op when unconfigured."""
    global _ENV_RESOLVED  # noqa: PLW0603
    if _ENV_RESOLVED:
        return
    profile = region_profile_from_env()
    _ENV_RESOLVED = True
    if profile is not None:
        register_profile(profile, default=True)


def get_profiles() -> dict[str, DetectionProfile]:
    """Every registered profile, keyed by ``name``."""
    ensure_env_region_profile()
    return dict(_REGISTRY)


def get_profile(name: str) -> DetectionProfile | None:
    """The registered profile called ``name``, or ``None``."""
    ensure_env_region_profile()
    return _REGISTRY.get(name)


def get_default_profile_name() -> str | None:
    """The ``name`` of the default profile, or ``None`` if none registered."""
    ensure_env_region_profile()
    return _DEFAULT_NAME


def get_active_region_profile() -> DetectionProfile | None:
    """The deployment's active region profile, or ``None`` when region
    detection is not configured (the neutral default)."""
    ensure_env_region_profile()
    return _REGISTRY.get(_DEFAULT_NAME) if _DEFAULT_NAME is not None else None


def region_profile_or_neutral() -> DetectionProfile:
    """The active profile, or a neutral one (no detector model, generic
    name) for call sites that only need profile-independent identity
    strings — e.g. the ``human`` detector name stamped on manual edits."""
    from src.config import DetectionProfile

    active = get_active_region_profile()
    return active if active is not None else DetectionProfile(name='region')


def _reset_registry_for_tests() -> None:
    """Test-only escape hatch — the module-level registry otherwise leaks
    across test cases. The next accessor call re-resolves the env."""
    global _DEFAULT_NAME, _ENV_RESOLVED  # noqa: PLW0603
    _REGISTRY.clear()
    _DEFAULT_NAME = None
    _ENV_RESOLVED = False


__all__ = [
    'REGION_DETECTION_ENV_PREFIX',
    'ensure_env_region_profile',
    'get_active_region_profile',
    'get_default_profile_name',
    'get_profile',
    'get_profiles',
    'region_profile_from_env',
    'region_profile_or_neutral',
    'register_profile',
]
