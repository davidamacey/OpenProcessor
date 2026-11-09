"""Read-only starter templates shipped under ``examples/<kind>/*.json``, and the
"is anything active" test, shared by the four config kinds (prompt packs, region
profiles, ingest profiles, open-vocab sets)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from pathlib import Path

    from src.services.config_store.snapshot import AxisRef


def template_names(templates_dir: Path) -> dict[str, Path]:
    """Template name (file stem) to path; empty when the directory is absent."""
    if not templates_dir.is_dir():
        return {}
    return {p.stem: p for p in sorted(templates_dir.glob('*.json'))}


def load_template(templates_dir: Path, name: str) -> dict[str, Any] | None:
    """The template body, or ``None`` when it is missing or unreadable."""
    path = template_names(templates_dir).get(name)
    if path is None:
        return None
    try:
        typed: dict[str, Any] | None = json.loads(path.read_text(encoding='utf-8'))
        return typed
    except (OSError, ValueError):
        return None


def active_ref(ref: AxisRef) -> tuple[str | None, int | None]:
    """``(name, revision)`` of the active entry; ``(None, None)`` when none or ``'off'``."""
    if isinstance(ref, tuple):
        return ref
    return None, None
