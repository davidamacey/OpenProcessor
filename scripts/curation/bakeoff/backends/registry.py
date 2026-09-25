"""Detector backend registry: name -> factory building a :class:`Detector`.

The five generic backends register from ``factory.py`` the first time the
registry is read. Anything else -- a backend for one domain's public models,
say -- lives in a plugin module named by a profile's ``backend_modules``;
:func:`load_backend_plugins` imports those modules and they call
:func:`register_backend` at import time. Mirrors the dataset-converter
registry in ``datasets.py``.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable

    from scripts.curation.bakeoff.profile import BakeoffProfile

    from .base import Detector

    BackendFactory = Callable[[argparse.Namespace], Detector]


_REGISTRY: dict[str, BackendFactory] = {}


def _ensure_builtins() -> None:
    # Imported for its side effect: factory.py registers the built-in backends.
    from . import factory  # noqa: F401


def register_backend(name: str, factory: BackendFactory, *, replace: bool = False) -> None:
    """Register ``factory`` under ``name`` (error on a clash unless ``replace``).

    Re-registering the same factory object is a no-op, so a plugin module
    imported twice is harmless.
    """
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not factory and not replace:
        raise ValueError(f'bake-off backend {name!r} already registered')
    _REGISTRY[name] = factory


def registered_backends() -> list[str]:
    """Every registered backend name (built-ins plus loaded plugins), sorted."""
    _ensure_builtins()
    return sorted(_REGISTRY)


def get_backend(name: str) -> BackendFactory:
    """The factory for ``name``; ``SystemExit`` listing the known names otherwise."""
    _ensure_builtins()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise SystemExit(
            f'unknown bake-off backend {name!r}; registered: {", ".join(sorted(_REGISTRY))} '
            "(domain backends come from a profile's backend_modules)"
        ) from None


def load_backend_plugins(profile: BakeoffProfile) -> None:
    """Import the profile's ``backend_modules`` so they self-register."""
    for module in profile.backend_modules:
        importlib.import_module(module)


__all__ = [
    'get_backend',
    'load_backend_plugins',
    'register_backend',
    'registered_backends',
]
