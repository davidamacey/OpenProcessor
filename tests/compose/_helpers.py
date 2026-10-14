"""Shared loaders for the compose contract tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_PATH = REPO_ROOT / 'docker-compose.yml'


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return yaml.safe_load(fh)


def _load_compose() -> dict[str, Any]:
    return _load_yaml(COMPOSE_PATH)


def _services() -> dict[str, dict[str, Any]]:
    return _load_compose()['services']
