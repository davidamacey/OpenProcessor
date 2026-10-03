"""``tests/test_full_system.py`` deletes OpenSearch indexes; it must never do
so without an explicit opt-in, and only under a safe prefix."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def system_test(monkeypatch: pytest.MonkeyPatch) -> Any:
    spec = importlib.util.spec_from_file_location(
        '_full_system_under_guard_test', Path(__file__).parent / 'test_full_system.py'
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv('OP_TEST_ALLOW_INDEX_DELETE', raising=False)
    monkeypatch.delenv('OP_TEST_INDEX_PREFIX', raising=False)
    return module


@pytest.fixture
def deletes(system_test: Any, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    sent: list[str] = []

    class _Response:
        status_code = 200

    def _delete(url: str, **_kw: Any) -> _Response:
        sent.append(url)
        return _Response()

    monkeypatch.setattr(system_test.requests, 'delete', _delete)
    return sent


def test_without_the_opt_in_no_delete_is_sent(system_test: Any, deletes: list[str]) -> None:
    assert system_test.clear_opensearch_data() is True
    assert deletes == []


def test_with_the_opt_in_only_the_prefix_is_deleted(
    system_test: Any, deletes: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_TEST_ALLOW_INDEX_DELETE', '1')
    monkeypatch.setenv('OP_TEST_INDEX_PREFIX', 'scratch_run_')
    assert system_test.clear_opensearch_data() is True
    assert [u.rsplit('/', 1)[-1] for u in deletes] == ['scratch_run_*']


@pytest.mark.parametrize('prefix', ['', '*', 'op_', 'op_prj_', 'visual*'])
def test_an_unsafe_prefix_is_refused_even_with_the_opt_in(
    system_test: Any, deletes: list[str], monkeypatch: pytest.MonkeyPatch, prefix: str
) -> None:
    monkeypatch.setenv('OP_TEST_ALLOW_INDEX_DELETE', '1')
    monkeypatch.setenv('OP_TEST_INDEX_PREFIX', prefix)
    assert system_test.clear_opensearch_data() is False
    assert deletes == []
