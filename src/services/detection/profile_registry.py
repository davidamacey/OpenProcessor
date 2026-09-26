"""In-memory registry of region :class:`~src.config.DetectionProfile`
instances, and resolution of the deployment's *active* region profile.

**Neutral by default.** An unconfigured deployment registers no profile:
:func:`get_active_region_profile` returns ``None``, ``GET /curation/methods``
advertises an empty ``detection_profile`` axis, and region detection (the
detection worker's cascade) stays off. Consumers must treat ``None`` as
"no region detection configured" and degrade cleanly.

The app ships with **no built-in region profile** — a different domain
(a license plate, a barcode, a manufacturing defect, …) is configured
purely through the environment:

- ``OP_REGION_PROFILE_PATH=<path>`` loads a profile from a JSON file (see
  :func:`region_profile_from_file` and ``examples/region_profiles/`` for
  a worked example). This is the normal way to select a non-trivial
  profile; the repo ships no profile data outside ``examples/``.
- ``OP_REGION_PROFILE=<name>`` selects a profile a deployment's own
  startup code already registered via :func:`register_profile`. An
  unknown name raises ``ValueError``. It does **not** resolve any
  built-in profile — there isn't one.
- ``OP_REGION_DETECTION_<FIELD>`` overrides individual fields on top of
  whichever profile ``OP_REGION_PROFILE_PATH`` / ``OP_REGION_PROFILE``
  selected (e.g. ``OP_REGION_DETECTION_SEGMENTER_TEXT_PROMPT``,
  ``OP_REGION_DETECTION_SECONDARY_SHAPE_GROUPS``). With neither set,
  these alone build a profile from the dataclass defaults (named by
  ``OP_REGION_DETECTION_NAME``, else ``region``).

The resolved profile is :func:`register_profile`'d as the default, so it
is exactly what ``GET /methods`` advertises. The ingest item detectors
are configured separately (``OP_INGEST_PRIMARY_*`` /
``OP_INGEST_SECONDARY_*``, ``routers/curation/ingest.py``). A leftover
retired ``OP_DETECTION_*`` var fails resolution loudly.

Further profiles (a deployment that detects more than one region type)
are added by startup code via :func:`register_profile`; every registered
profile is advertised and selectable by name.
"""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from src.config import DetectionProfile

REGION_DETECTION_ENV_PREFIX = 'OP_REGION_DETECTION_'

_REGISTRY: dict[str, DetectionProfile] = {}
_DEFAULT_NAME: str | None = None
_ENV_RESOLVED = False
_RESOLVE_LOCK = threading.Lock()


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


def region_profile_from_dict(data: dict[str, Any], *, source: str = 'file') -> DetectionProfile:
    """Build a :class:`DetectionProfile` from a flat ``{field_name: value}``
    mapping (the decoded JSON of a profile file).

    Underscore-prefixed keys (``_comment`` etc.) are ignored; any other
    unknown key is rejected, and ``name`` is required. Tuple fields
    (``letterbox_fill``, ``auto_confirm_aspect``, ``auto_confirm_area_frac``)
    and frozenset fields (``secondary_shape_groups``, ``class_ids``,
    ``parent_classes``, ``text_stopwords``, ``text_placeholders``) are given
    as JSON lists. ``source`` names the input in error messages. Raises on
    anything malformed -- a typo must fail loudly, not silently fall back
    to a default.
    """
    from dataclasses import fields as dc_fields

    from src.config import DetectionProfile

    if not isinstance(data, dict):
        msg = f'region profile {source!r} must be a JSON object'
        raise ValueError(msg)
    values = {k: v for k, v in data.items() if not k.startswith('_')}
    if 'name' not in values:
        msg = f'region profile {source!r} is missing the required "name" field'
        raise ValueError(msg)

    field_by_name = {f.name: f for f in dc_fields(DetectionProfile)}
    kwargs: dict[str, object] = {}
    for key, value in values.items():
        f = field_by_name.get(key)
        if f is None:
            msg = f'region profile {source!r}: unknown field {key!r}'
            raise ValueError(msg)
        annotation = str(f.type)
        if annotation.startswith('tuple'):
            kwargs[key] = tuple(value)
        elif annotation.startswith('frozenset'):
            kwargs[key] = frozenset(value)
        else:
            kwargs[key] = value
    return DetectionProfile(**kwargs)  # type: ignore[arg-type]


def region_profile_from_file(path: str) -> DetectionProfile:
    """Load a :class:`DetectionProfile` from a JSON file.

    The file is a flat object of ``{field_name: value}`` pairs, decoded by
    :func:`region_profile_from_dict`. See
    ``examples/region_profiles/license_plate.json`` (text-reading) and
    ``examples/region_profiles/vehicle_wheel.json`` (text-free,
    segmenter-only) for worked examples.
    """
    import json
    from pathlib import Path

    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    return region_profile_from_dict(raw, source=path)


def region_profile_from_env() -> DetectionProfile | None:
    """Resolve the region profile the environment asks for (see module
    docstring), without registering it. ``None`` when nothing is configured.
    """
    from src.config import DetectionProfile
    from src.config.detection_profile import reject_legacy_detection_env

    reject_legacy_detection_env()
    path = os.environ.get('OP_REGION_PROFILE_PATH', '').strip()
    name = os.environ.get('OP_REGION_PROFILE', '').strip()
    has_overrides = DetectionProfile.env_overrides_present(REGION_DETECTION_ENV_PREFIX)
    if not path and not name and not has_overrides:
        return None
    base: DetectionProfile | None = None
    if path:
        base = region_profile_from_file(path)
    elif name:
        base = _REGISTRY.get(name)
        if base is None:
            known = sorted(_REGISTRY)
            msg = (
                f'OP_REGION_PROFILE={name!r} is not a registered profile (no profile ships '
                'built in -- set OP_REGION_PROFILE_PATH to a profile file instead, e.g. '
                f'examples/region_profiles/license_plate.json); known registered: {known}'
            )
            raise ValueError(msg)
    return DetectionProfile.from_env(REGION_DETECTION_ENV_PREFIX, name='region', base=base)


def ensure_env_region_profile() -> None:
    """Resolve the env-configured region profile once per process and
    register it as the default. Idempotent; a no-op when unconfigured."""
    global _ENV_RESOLVED  # noqa: PLW0603
    if _ENV_RESOLVED:
        return
    # Sync routes run in a thread pool: without the lock (and with the flag
    # set before registration) a concurrent first lookup saw "resolved" but
    # no profile yet and answered 409 no-profile on a configured deployment.
    with _RESOLVE_LOCK:
        if _ENV_RESOLVED:
            return
        profile = region_profile_from_env()
        if profile is not None:
            register_profile(profile, default=True)
        _ENV_RESOLVED = True


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
    'region_profile_from_dict',
    'region_profile_from_env',
    'region_profile_from_file',
    'region_profile_or_neutral',
    'register_profile',
]
