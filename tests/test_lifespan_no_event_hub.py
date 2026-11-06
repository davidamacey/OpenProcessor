"""The event hub and its JSONL log are gone from the app's source and lifespan."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.config.retired_env import RETIRED_ENV, reject_retired_env


SRC = Path(__file__).resolve().parents[1] / 'src'


def _imports(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f'{node.module}.{a.name}' for a in node.names)
        elif isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
    return names


def test_no_module_imports_the_event_hub() -> None:
    offenders = [
        str(p.relative_to(SRC))
        for p in SRC.rglob('*.py')
        if any(name.endswith('event_hub') for name in _imports(p))
    ]
    assert offenders == []
    assert not (SRC / 'services/curation/event_hub.py').exists()


def test_the_lifespan_starts_the_listener_and_has_no_jsonl_path() -> None:
    text = (SRC / 'app/lifespan.py').read_text()
    assert 'start_listener' in text
    assert 'stop_listener' in text
    assert 'jsonl' not in text.lower()
    assert (
        'to_thread' not in text.split('async def _start_event_listener')[1].split('\nasync def ')[0]
    )


@pytest.mark.parametrize('name', ['OP_EVENT_BUS', 'OP_EVENT_LOG_MAX_BYTES'])
def test_the_event_log_settings_are_retired(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    assert RETIRED_ENV[name].startswith('removed:')
    monkeypatch.setenv(name, '1')
    with pytest.raises(RuntimeError, match=name):
        reject_retired_env()
