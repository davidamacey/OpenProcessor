"""Where ``default``'s per-project dirs resolve under a test's env roots.

``default`` is an ordinary project (no special-casing), so its resources
come from ``src.config.projects.resources_for_new`` like every other
project's: exports live under ``OP_PROJECTS_DATA_ROOT/default/exports``. Tests that
point those env vars at ``tmp_path`` use these helpers to build the
expected paths instead of hard-coding either layout.
"""

from __future__ import annotations

from pathlib import Path

from src.config.projects import DEFAULT_SLUG


def default_export_root(data_root: Path, *, create: bool = True) -> Path:
    """``default``'s ``export_root`` when ``OP_PROJECTS_DATA_ROOT=data_root``."""
    path = Path(data_root) / DEFAULT_SLUG / 'exports'
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
