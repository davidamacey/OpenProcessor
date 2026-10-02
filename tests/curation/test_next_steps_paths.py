"""Every follow-up a job report serves must be a real, project-scoped route.

Found live: the combine report offered ``POST /clusters/train``, an unscoped
core route that 404s when a client calls it under the project mount.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.services.curation import next_steps


ROOT = Path(__file__).resolve().parents[2]
PROJECT_MOUNT = '/curation/projects/{project}'
OPENAPI = json.loads((ROOT / 'contracts/openapi/curation.json').read_text())


@pytest.mark.parametrize('build', next_steps.ALL_STEPS, ids=lambda b: b.__name__)
def test_served_step_resolves_in_the_published_openapi(build) -> None:
    step = build()
    full = PROJECT_MOUNT + step['path']
    assert step['path'].startswith('/')
    assert step['method'].lower() in OPENAPI['paths'].get(full, {}), full


def test_every_step_builder_is_registered() -> None:
    builders = {
        name
        for name, fn in vars(next_steps).items()
        if callable(fn) and not name.startswith('_') and fn.__module__ == next_steps.__name__
    }
    assert builders == {b.__name__ for b in next_steps.ALL_STEPS}


def test_no_producer_hand_writes_a_step() -> None:
    offenders = [
        str(p.relative_to(ROOT))
        for p in (ROOT / 'src').rglob('*.py')
        if p.name != 'next_steps.py'
        and "'path': '/" in p.read_text()
        and "'reason':" in p.read_text()
    ]
    assert offenders == []
