"""Coverage for ``src/services/curation/autolabel/cli.py``, a
zero-coverage subprocess entry point.

This module has no ``argparse`` surface — its input is a ``state.json`` file
written by :func:`src.services.curation.autolabel.job.start_job`, not
CLI flags. Tests exercise the module's actual shape instead: pipeline
path resolution, the ``_run_pipeline`` state-machine (success, each
failure branch, cancellation), and ``main()``'s exit-code mapping.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import src.services.curation.autolabel.cli as cli_mod
from src.services.curation.autolabel.job import _JobState


# ---------------------------------------------------------------------------
# _resolve_pipeline_fn — module:qualname -> callable
# ---------------------------------------------------------------------------


def test_resolve_pipeline_fn_finds_a_real_callable() -> None:
    fn = cli_mod._resolve_pipeline_fn('math:sqrt')
    import math

    assert fn is math.sqrt


def test_resolve_pipeline_fn_resolves_nested_attribute_path() -> None:
    fn = cli_mod._resolve_pipeline_fn('src.services.curation.autolabel.job:_JobState')
    assert fn is _JobState


@pytest.mark.parametrize('bad_path', ['', 'no_colon_here', ':missing_module', 'os:'])
def test_resolve_pipeline_fn_rejects_malformed_paths(bad_path: str) -> None:
    with pytest.raises(ValueError, match='malformed pipeline path'):
        cli_mod._resolve_pipeline_fn(bad_path)


def test_resolve_pipeline_fn_raises_on_unimportable_module() -> None:
    with pytest.raises(ImportError):
        cli_mod._resolve_pipeline_fn('no_such_module_xyz:fn')


def test_resolve_pipeline_fn_raises_on_missing_attribute() -> None:
    with pytest.raises(AttributeError):
        cli_mod._resolve_pipeline_fn('math:no_such_attr_xyz')


# ---------------------------------------------------------------------------
# _write_exit_code
# ---------------------------------------------------------------------------


def test_write_exit_code_writes_the_code(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    exit_file = tmp_path / 'exit_code'
    monkeypatch.setattr(cli_mod, '_EXIT_CODE_FILE', exit_file)
    cli_mod._write_exit_code(130)
    assert exit_file.read_text() == '130'


def test_write_exit_code_suppresses_oserror(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    # A directory in place of the file: write_text raises OSError (IsADirectoryError).
    exit_dir = tmp_path / 'exit_code'
    exit_dir.mkdir()
    monkeypatch.setattr(cli_mod, '_EXIT_CODE_FILE', exit_dir)
    cli_mod._write_exit_code(1)  # must not raise


# ---------------------------------------------------------------------------
# _run_pipeline — the state machine, one branch per outcome
# ---------------------------------------------------------------------------


def _patch_state_io(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, state: _JobState
) -> list[dict[str, Any]]:
    """Route _read_state/_atomic_write/lock-file cleanup through
    in-memory/tmp-path doubles so no test touches the real state dir."""
    writes: list[dict[str, Any]] = []
    monkeypatch.setattr(cli_mod, '_read_state', lambda: state)
    monkeypatch.setattr(cli_mod, '_atomic_write', writes.append)
    monkeypatch.setattr(cli_mod, '_RUNNING_LOCK', tmp_path / 'running.lock')
    monkeypatch.setattr(cli_mod, '_CANCEL_FLAG', tmp_path / 'cancel.flag')
    return writes


@pytest.mark.asyncio
async def test_run_pipeline_no_pipeline_path_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    state = _JobState(pipeline='')
    writes = _patch_state_io(monkeypatch, tmp_path, state)

    rc = await cli_mod._run_pipeline()

    assert rc == 1
    assert state.status == 'failed'
    assert state.error is not None
    assert 'no pipeline import path' in state.error
    assert writes  # state was persisted


@pytest.mark.asyncio
async def test_run_pipeline_unresolvable_pipeline_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    state = _JobState(pipeline='no_such_module_xyz:fn')
    _patch_state_io(monkeypatch, tmp_path, state)

    rc = await cli_mod._run_pipeline()

    assert rc == 1
    assert state.status == 'failed'
    assert state.error is not None
    assert 'cannot resolve pipeline' in state.error


@pytest.mark.asyncio
async def test_run_pipeline_opensearch_init_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    state = _JobState(pipeline='math:sqrt')
    _patch_state_io(monkeypatch, tmp_path, state)

    async def _boom() -> None:
        raise RuntimeError('no opensearch')

    monkeypatch.setattr(cli_mod, '_build_opensearch', _boom)

    rc = await cli_mod._run_pipeline()

    assert rc == 1
    assert state.status == 'failed'
    assert state.error is not None
    assert 'opensearch init failed' in state.error
    assert state.error_detail


@pytest.mark.asyncio
async def test_run_pipeline_success_records_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    state = _JobState(pipeline='fake:pipeline', args={'foo': 'bar'})
    _patch_state_io(monkeypatch, tmp_path, state)
    monkeypatch.setattr(cli_mod, '_build_opensearch', _fake_build_opensearch)

    captured_kwargs: dict[str, Any] = {}

    async def _fake_pipeline(*, opensearch: Any, progress: Any, **kwargs: Any) -> dict[str, Any]:
        captured_kwargs.update(kwargs)
        assert opensearch == 'fake-os'
        assert isinstance(progress, cli_mod._Progress)
        return {'promoted': 3}

    monkeypatch.setattr(cli_mod, '_resolve_pipeline_fn', lambda _path: _fake_pipeline)

    rc = await cli_mod._run_pipeline()

    assert rc == 0
    assert state.status == 'completed'
    assert state.result == {'promoted': 3}
    assert captured_kwargs == {'foo': 'bar'}


@pytest.mark.asyncio
async def test_run_pipeline_non_dict_result_wrapped_as_raw(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    state = _JobState(pipeline='fake:pipeline')
    _patch_state_io(monkeypatch, tmp_path, state)
    monkeypatch.setattr(cli_mod, '_build_opensearch', _fake_build_opensearch)

    async def _fake_pipeline(*, opensearch: Any, progress: Any, **_kw: Any) -> int:
        return 42

    monkeypatch.setattr(cli_mod, '_resolve_pipeline_fn', lambda _path: _fake_pipeline)

    rc = await cli_mod._run_pipeline()

    assert rc == 0
    assert state.result == {'raw': '42'}


@pytest.mark.asyncio
async def test_run_pipeline_cancelled_returns_130(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    state = _JobState(pipeline='fake:pipeline')
    _patch_state_io(monkeypatch, tmp_path, state)
    monkeypatch.setattr(cli_mod, '_build_opensearch', _fake_build_opensearch)

    async def _fake_pipeline(**_kw: Any) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(cli_mod, '_resolve_pipeline_fn', lambda _path: _fake_pipeline)

    rc = await cli_mod._run_pipeline()

    assert rc == 130
    assert state.status == 'cancelled'


@pytest.mark.asyncio
async def test_run_pipeline_generic_exception_fails_with_detail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    state = _JobState(pipeline='fake:pipeline')
    _patch_state_io(monkeypatch, tmp_path, state)
    monkeypatch.setattr(cli_mod, '_build_opensearch', _fake_build_opensearch)

    async def _fake_pipeline(**_kw: Any) -> None:
        raise ValueError('bad embedding shape')

    monkeypatch.setattr(cli_mod, '_resolve_pipeline_fn', lambda _path: _fake_pipeline)

    rc = await cli_mod._run_pipeline()

    assert rc == 1
    assert state.status == 'failed'
    assert state.error is not None
    assert 'bad embedding shape' in state.error
    assert state.error_detail


@pytest.mark.asyncio
async def test_run_pipeline_cleans_up_lock_and_cancel_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    state = _JobState(pipeline='fake:pipeline')
    lock = tmp_path / 'running.lock'
    flag = tmp_path / 'cancel.flag'
    lock.touch()
    flag.touch()
    monkeypatch.setattr(cli_mod, '_read_state', lambda: state)
    monkeypatch.setattr(cli_mod, '_atomic_write', lambda _payload: None)
    monkeypatch.setattr(cli_mod, '_RUNNING_LOCK', lock)
    monkeypatch.setattr(cli_mod, '_CANCEL_FLAG', flag)
    monkeypatch.setattr(cli_mod, '_build_opensearch', _fake_build_opensearch)

    async def _fake_pipeline(**_kw: Any) -> dict[str, Any]:
        return {}

    monkeypatch.setattr(cli_mod, '_resolve_pipeline_fn', lambda _path: _fake_pipeline)

    await cli_mod._run_pipeline()

    assert not lock.exists()
    assert not flag.exists()


async def _fake_build_opensearch() -> str:
    return 'fake-os'


# ---------------------------------------------------------------------------
# main() — the sync entry point's exit-code mapping
# ---------------------------------------------------------------------------


def test_main_returns_amain_result_and_writes_exit_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.setattr(cli_mod, '_STATE_FILE', tmp_path / 'jobs' / 'state.json')
    exit_file = tmp_path / 'exit_code'
    monkeypatch.setattr(cli_mod, '_EXIT_CODE_FILE', exit_file)

    async def _fake_amain() -> int:
        return 0

    monkeypatch.setattr(cli_mod, '_amain', _fake_amain)

    rc = cli_mod.main()

    assert rc == 0
    assert exit_file.read_text() == '0'


def test_main_keyboard_interrupt_returns_130(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.setattr(cli_mod, '_STATE_FILE', tmp_path / 'jobs' / 'state.json')
    exit_file = tmp_path / 'exit_code'
    monkeypatch.setattr(cli_mod, '_EXIT_CODE_FILE', exit_file)

    async def _fake_amain() -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_mod, '_amain', _fake_amain)

    rc = cli_mod.main()

    assert rc == 130
    assert exit_file.read_text() == '130'


def test_main_fatal_exception_writes_failed_state_and_returns_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.setattr(cli_mod, '_STATE_FILE', tmp_path / 'jobs' / 'state.json')
    exit_file = tmp_path / 'exit_code'
    monkeypatch.setattr(cli_mod, '_EXIT_CODE_FILE', exit_file)

    async def _fake_amain() -> int:
        raise RuntimeError('event loop exploded')

    monkeypatch.setattr(cli_mod, '_amain', _fake_amain)

    state = _JobState()
    written: list[dict[str, Any]] = []
    monkeypatch.setattr(cli_mod, '_read_state', lambda: state)
    monkeypatch.setattr(cli_mod, '_atomic_write', written.append)

    rc = cli_mod.main()

    assert rc == 1
    assert exit_file.read_text() == '1'
    assert state.status == 'failed'
    assert state.error is not None
    assert 'event loop exploded' in state.error
    assert written
