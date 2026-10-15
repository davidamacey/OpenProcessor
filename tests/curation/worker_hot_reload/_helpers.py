"""Shared builders for the worker hot-reload tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from src.config.curation_accessors import base_curation_config
from src.config.projects import ProjectRecord, resources_for_new
from src.services.config_store import get_global_config_store


def _record(slug: str, revision: int = 1) -> ProjectRecord:
    from datetime import UTC, datetime

    now = datetime.now(UTC).isoformat()
    return ProjectRecord(
        slug=slug,
        display_name=slug,
        description='',
        status='active',
        revision=revision,
        created_at=now,
        updated_at=now,
        origin=None,
        resources=resources_for_new(slug, base_curation_config()),
    )


# =============================================================================
# build_runtime / quiesce_and_swap
# =============================================================================


def _fake_args() -> Any:
    ns = MagicMock()
    ns.segmenter_url = ''
    return ns


def _fake_ctors() -> dict[str, Any]:
    """Stand-ins for the four heavy-IO constructors ``build_runtime``
    now takes as parameters (never imports fresh) -- mirrors how
    ``runner.py`` passes its own module-level names, which is exactly
    what lets a real monkeypatch on those names reach a rebuild."""
    return {
        'region_detector_cls': MagicMock(),
        'ocr_recognizer_cls': MagicMock(),
        'segmenter_cls': MagicMock(),
        'build_vlm': MagicMock(),
    }


def _swap_kw() -> dict[str, Any]:
    """The extra collaborators a swap needs: the pinned deployment-wide
    registry store and the active-endpoint resolver (none active here)."""
    return {
        'registry': get_global_config_store(mode='pinned'),
        'get_active_vlm': lambda: None,
        **_fake_ctors(),
    }
