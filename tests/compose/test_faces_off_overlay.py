"""The faces-off overlay stays the base compose file minus the two face model preloads."""

from __future__ import annotations

from typing import Any

from compose._helpers import COMPOSE_PATH, REPO_ROOT, _load_yaml


OVERLAY = REPO_ROOT / 'docker-compose.faces-off.yml'
FACE_PRELOADS = {'--load-model=scrfd_10g_bnkps', '--load-model=arcface_w600k_r50'}


def _command(doc: dict[str, Any]) -> list[str]:
    return list(doc['services']['triton-server']['command'])


def test_the_overlay_command_is_the_base_command_without_the_face_models() -> None:
    base = _command(_load_yaml(COMPOSE_PATH))
    assert set(base) >= FACE_PRELOADS
    assert _command(_load_yaml(OVERLAY)) == [c for c in base if c not in FACE_PRELOADS]


def test_the_overlay_changes_nothing_but_the_triton_command() -> None:
    doc = _load_yaml(OVERLAY)
    assert set(doc['services']) == {'triton-server'}
    assert set(doc['services']['triton-server']) == {'command'}
