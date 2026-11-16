"""``contracts/json/retired_env.json`` says what ``reject_retired_env`` says, name by name.

The Rust server refuses to start from this file (``native/op_config``, module ``retired``), so its
refusal must read exactly as the Python guard's does, and the committed file must be the table's
current rendering.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from src.config.retired_env import RETIRED_ENV, reject_retired_env


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / 'contracts' / 'json' / 'retired_env.json'


def _generator() -> Any:
    sys.path.insert(0, str(ROOT / 'scripts' / 'codegen'))
    try:
        spec = importlib.util.spec_from_file_location(
            '_retired_env_contract', ROOT / 'scripts' / 'codegen' / 'retired_env_contract.py'
        )
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def _entries() -> list[dict[str, str]]:
    entries: list[dict[str, str]] = json.loads(CONTRACT.read_text(encoding='utf-8'))['retired']
    return entries


def test_the_committed_file_is_the_tables_rendering() -> None:
    assert CONTRACT.read_text(encoding='utf-8') == _generator().render()


def test_the_file_lists_every_retired_name_once_sorted() -> None:
    names = [e['name'] for e in _entries()]
    assert names == sorted(RETIRED_ENV)


@pytest.mark.parametrize('name', sorted(RETIRED_ENV))
def test_the_python_refusal_for_one_name_is_the_files_words(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = next(e for e in _entries() if e['name'] == name)
    monkeypatch.setenv(name, 'x')
    with pytest.raises(RuntimeError) as refusal:
        reject_retired_env()
    assert str(refusal.value) == f'retired env var(s) set: {entry["name"]} was {entry["was"]}'


def test_the_check_flag_fails_on_a_stale_file(tmp_path: Path) -> None:
    stale = tmp_path / 'retired_env.json'
    stale.write_text('{}\n', encoding='utf-8')
    assert _generator().run(stale, check_only=True) == 1
    assert _generator().run(stale, check_only=False) == 0
    assert _generator().run(stale, check_only=True) == 0
