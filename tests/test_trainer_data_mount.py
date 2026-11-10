"""The trainer publishes run artifacts (run.log, weights, metrics) through the rooted store under
``projects/<slug>/training/...``, which lives in ./data/projects; a read-only /app/data mount
made every publish fail with EROFS."""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_trainer_data_mount_is_writable() -> None:
    compose = yaml.safe_load((ROOT / 'docker-compose.yml').read_text(encoding='utf-8'))
    mounts = [
        v
        for v in compose['services']['curation-trainer']['volumes']
        if isinstance(v, str) and v.split(':')[1:2] == ['/app/data']
    ]
    assert mounts, 'curation-trainer has no /app/data mount'
    assert all(not m.endswith(':ro') for m in mounts), mounts
