"""The trainer image imports ``src.tasks.runtime`` and ``src.db``; every third-party package those
modules import at module level must be installed there, at the pin the api image uses."""

from __future__ import annotations

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _pins(rel: str) -> dict[str, str]:
    out = {}
    for line in (ROOT / rel).read_text(encoding='utf-8').splitlines():
        m = re.match(r'^([A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?==([^\s;#]+)', line)
        if m:
            out[m.group(1).lower().replace('_', '-')] = m.group(2)
    return out


@pytest.mark.parametrize(
    'package',
    [
        # src/db/pool_metrics.py (via src.db.engine) does `from prometheus_client import Histogram`
        'prometheus-client',
        # src/tasks/runtime/context.py imports tenacity.AsyncRetrying
        'tenacity',
    ],
)
def test_trainer_installs_runtime_imports_at_the_api_pin(package: str) -> None:
    trainer = _pins('docker/trainer/requirements.txt')
    api = _pins('requirements.txt')
    assert trainer.get(package) == api[package]
