"""The committed goldens of the native engine's geometry and routing ports are what the Python
functions answer today (``scripts/codegen/frame_engine_goldens.py``)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / 'native' / 'frame_engine' / 'tests' / 'golden'
SCRIPT = ROOT / 'scripts' / 'codegen' / 'frame_engine_goldens.py'


@pytest.fixture(scope='module')
def rendered() -> dict[str, str]:
    spec = importlib.util.spec_from_file_location('frame_engine_goldens', SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.render()


def test_the_committed_files_are_what_the_generator_writes(rendered: dict[str, str]) -> None:
    assert set(rendered) == {p.name for p in GOLDEN.iterdir()}
    for name, text in rendered.items():
        assert (GOLDEN / name).read_text(encoding='utf-8') == text, f'{name} is stale'


def test_the_goldens_cover_what_the_native_tests_rely_on(rendered: dict[str, str]) -> None:
    geometry = json.loads(rendered['geometry.json'])
    assert all(
        len(geometry[k]) >= 400 for k in ('letterbox', 'square', 'undo_letterbox', 'crop_rect')
    )
    weights = rendered['weights.jsonl'].splitlines()
    assert len(weights) > 10_000
    assert any('false' in line for line in weights), 'the plain branch'
    cases = json.loads(rendered['jpeg_header.json'])['cases']
    routes = {c['route']['kind'] + ':' + c['route']['reason'] for c in cases}
    assert {
        'gpu:', 'refused:corrupt', 'refused:huge', 'cpu:not_jpeg', 'cpu:precision',
        'cpu:components', 'cpu:progressive', 'cpu:truncated', 'cpu:small',
    } <= routes  # fmt: skip
