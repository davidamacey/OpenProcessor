"""The fit runs in a non-daemonic child process: launch, result, failure, cancel, timeout.

The real :mod:`fit_process` runs as a child; a copy of it sits next to a fake ``trainer`` module
(the child resolves ``trainer`` from its own script directory), so no Ultralytics is needed.
"""

from __future__ import annotations

import multiprocessing
import os
import shutil
import textwrap
import threading
from pathlib import Path

import fit_process
import job_spec
import pytest
from job_status import StatusState


REPO = Path(__file__).resolve().parents[2]


def _make_world(tmp_path, monkeypatch, fake_body: str):
    script_dir = tmp_path / 'child'
    script_dir.mkdir()
    shutil.copy(Path(fit_process.__file__), script_dir / 'fit_process.py')
    (script_dir / 'trainer.py').write_text(
        'import time\n'
        'def run_training(spec, state):\n' + textwrap.indent(textwrap.dedent(fake_body), '    ')
    )
    monkeypatch.setattr(fit_process, '_CHILD_SCRIPT', script_dir / 'fit_process.py')
    monkeypatch.setenv('PYTHONPATH', f'{REPO / "docker" / "trainer"}:{REPO}')
    monkeypatch.setattr(job_spec, 'TMP_ROOT', tmp_path / 'scratch')
    monkeypatch.setattr(fit_process, 'POLL_INTERVAL_S', 0.05)
    export = tmp_path / 'export'
    export.mkdir()
    spec = job_spec.parse_and_validate_spec(
        'job1', {'dataset_export_dir': str(export), 'model_family': 'yolo26', 'model_size': 'n'}
    )
    return spec, StatusState(job_id='job1')


def test_the_childs_state_and_output_reach_the_parent(tmp_path, monkeypatch, capsys) -> None:
    spec, state = _make_world(
        tmp_path,
        monkeypatch,
        """
        print('[fake] hello from the child')
        with state.lock:
            state.state = 'finished'
            state.current_epoch = 3
            state.eval = {'map50': 0.4}
            state.checkpoint_path = '/x/weights/best.pt'
        """,
    )
    fit_process.run_fit(spec, state)
    assert (state.state, state.current_epoch, state.eval) == ('finished', 3, {'map50': 0.4})
    assert state.checkpoint_path == '/x/weights/best.pt'
    assert '[fake] hello from the child' in capsys.readouterr().out


def test_a_failing_fit_raises_with_the_childs_error(tmp_path, monkeypatch) -> None:
    spec, state = _make_world(tmp_path, monkeypatch, "raise RuntimeError('CUDA exploded')\n")
    with pytest.raises(fit_process.FitProcessError, match='CUDA exploded'):
        fit_process.run_fit(spec, state)
    assert state.state == 'failed'


def test_a_crash_without_a_result_is_a_failure(tmp_path, monkeypatch) -> None:
    spec, state = _make_world(tmp_path, monkeypatch, 'import os\nos._exit(7)\n')
    with pytest.raises(fit_process.FitProcessError, match='code 7'):
        fit_process.run_fit(spec, state)


def test_a_cancel_reaches_the_child_and_ends_the_run(tmp_path, monkeypatch) -> None:
    spec, state = _make_world(
        tmp_path,
        monkeypatch,
        """
        with state.lock:
            state.state = 'running'
        assert state.cancel_requested.wait(30), 'cancel never arrived'
        with state.lock:
            state.state = 'cancelled'
        """,
    )
    threading.Timer(1.0, state.cancel_requested.set).start()
    fit_process.run_fit(spec, state)
    assert state.state == 'cancelled'


def test_the_wall_clock_bound_kills_the_child(tmp_path, monkeypatch) -> None:
    spec, state = _make_world(
        tmp_path, monkeypatch, 'import os\nprint(os.getpid(), flush=True)\ntime.sleep(300)\n'
    )
    monkeypatch.setattr(fit_process, 'FIT_TIMEOUT_S', 1.0)
    monkeypatch.setattr(fit_process, 'KILL_GRACE_S', 5.0)
    pids: list[int] = []
    real_popen = fit_process.subprocess.Popen

    def spy(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        pids.append(proc.pid)
        return proc

    monkeypatch.setattr(fit_process.subprocess, 'Popen', spy)
    with pytest.raises(fit_process.FitProcessError, match='exceeded'):
        fit_process.run_fit(spec, state)
    with pytest.raises(ProcessLookupError):
        os.kill(pids[0], 0)


def _run_from_daemonic(spec, state, result) -> None:
    try:
        fit_process.run_fit(spec, state)
        result.put(state.state)
    except Exception as exc:  # the assertion below reports it
        result.put(f'{type(exc).__name__}: {exc}')


def test_the_fit_can_start_dataloader_workers_under_a_daemonic_parent(
    tmp_path, monkeypatch
) -> None:
    """A Celery prefork child is daemonic; the fit must still be able to have children."""
    spec, state = _make_world(
        tmp_path,
        monkeypatch,
        """
        import multiprocessing
        p = multiprocessing.Process(target=time.sleep, args=(0.1,))
        p.start()
        p.join()
        with state.lock:
            state.state = 'finished'
        """,
    )
    ctx = multiprocessing.get_context('fork')
    result = ctx.Queue()
    proc = ctx.Process(target=_run_from_daemonic, args=(spec, state, result), daemon=True)
    proc.start()
    proc.join(60)
    assert result.get(timeout=5) == 'finished'
