"""In-memory registry of configured :class:`~src.config.DetectionProfile`
instances.

Today exactly one profile is ever constructed at runtime — the module-level
``REFERENCE_LICENSE_PLATE_PROFILE`` in :mod:`src.services.detection.cascade_detect`, which
registers itself as the default the moment that module is imported. This
module exists so ``GET /curation/methods``' ``detection_profile`` axis (see
``src.services.curation.strategy_registry``) has a real mechanism to read
from rather than a single hardcoded entry: a future deployment that wants to
detect more than one region type (a license plate AND a shipping label,
say) registers a second :class:`~src.config.DetectionProfile` here — no
database, no extra service, just a process-lifetime dict a deployment's own
startup code (or a future config-driven loader) populates via
:func:`register_profile`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from src.config import DetectionProfile

_REGISTRY: dict[str, DetectionProfile] = {}
_DEFAULT_NAME: str | None = None


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


def get_profiles() -> dict[str, DetectionProfile]:
    """Every registered profile, keyed by ``name``."""
    return dict(_REGISTRY)


def get_default_profile_name() -> str | None:
    """The ``name`` of the default profile, or ``None`` if none registered yet."""
    return _DEFAULT_NAME


def _reset_registry_for_tests() -> None:
    """Test-only escape hatch — the module-level registry otherwise leaks
    across test cases."""
    global _DEFAULT_NAME  # noqa: PLW0603
    _REGISTRY.clear()
    _DEFAULT_NAME = None


__all__ = [
    'get_default_profile_name',
    'get_profiles',
    'register_profile',
]
