"""Run one fit in its own non-daemonic process, so Ultralytics can start DataLoader workers.

Celery's prefork pool children are daemonic, and :mod:`multiprocessing` forbids a daemonic
process from having children (https://docs.celeryq.dev/en/stable/userguide/workers.html and
https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Process.daemon); that
broke ``workers>0`` and the per-class validation pass. A :mod:`subprocess` child is an ordinary
OS process and is not subject to the rule, so the task keeps the prefork pool and its fresh
child per run (design 02b) and launches this module as ``python fit_process.py <control dir>``.

Parent and child share a control directory under the run's scratch root:

* ``spec.json``   the run's raw spec (parent writes, child reads)
* ``state.json``  the child's :class:`~job_status.StatusState` snapshot, rewritten atomically
* ``cancel``      created by the parent when the run's cancel was requested; the child sets
                  ``state.cancel_requested``, which the epoch callback reads

The child's output is piped to the parent's stdout so the run log tee still sees it. The child is
a session leader that sets ``PR_SET_PDEATHSIG`` (SIGTERM when the parent dies) at start, and the parent
kills its process group on every exit path, so a revoked task leaves no orphan training on the GPU.
"""

from __future__ import annotations

import contextlib
import ctypes
import dataclasses
import json
import os
import signal
import subprocess  # nosec B404 - launches this repo's own module
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from job_spec import parse_and_validate_spec
from job_status import StatusState
from logutil import get_logger


if TYPE_CHECKING:
    from job_spec import JobSpec

logger = get_logger('trainer.fit_process')

POLL_INTERVAL_S = 1.0
_CHILD_SCRIPT = Path(__file__).resolve()
KILL_GRACE_S = 10.0
# Wall-clock bound on one fit; a run past it is killed and fails.
FIT_TIMEOUT_S = 7 * 24 * 3600.0
_SNAPSHOT_SKIP = frozenset({'job_id', 'cancel_requested', 'lock'})
_PR_SET_PDEATHSIG = 1


class FitProcessError(RuntimeError):
    """The fit process ended without a result: non-zero exit, a kill or the wall-clock bound."""


def _control_dir(spec: JobSpec) -> Path:
    return spec.tmp_root / 'fit'


def _snapshot(state: StatusState) -> dict[str, Any]:
    with state.lock:
        return {
            f.name: getattr(state, f.name)
            for f in dataclasses.fields(state)
            if f.name not in _SNAPSHOT_SKIP
        }


def _write_snapshot(path: Path, state: StatusState) -> None:
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(_snapshot(state), default=str), encoding='utf-8')
    tmp.replace(path)


def _apply_snapshot(path: Path, state: StatusState) -> None:
    """Copy the child's snapshot into the parent's state; an unreadable file changes nothing."""
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return
    names = {f.name for f in dataclasses.fields(state)} - _SNAPSHOT_SKIP
    with state.lock:
        for key, value in data.items():
            if key in names:
                setattr(state, key, value)


def _die_with_parent() -> None:
    """Child side: get SIGTERM when the parent dies."""
    with contextlib.suppress(OSError, AttributeError):
        ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_PDEATHSIG, signal.SIGTERM)


def _stop_group(proc: subprocess.Popen[str]) -> None:
    """SIGTERM the child's process group, SIGKILL it after the grace period."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, sig)
        try:
            proc.wait(timeout=KILL_GRACE_S)
        except subprocess.TimeoutExpired:
            continue
        return


def _pump_output(proc: subprocess.Popen[str]) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()


def run_fit(spec: JobSpec, state: StatusState) -> None:
    """Train in a child process; blocking, called from the task's fit thread.

    Mirrors :func:`trainer.run_training`'s contract: returns when the run finished or was
    cancelled (``state.state`` says which), raises for any other failure.
    """
    ctl = _control_dir(spec)
    ctl.mkdir(parents=True, exist_ok=True)
    (ctl / 'spec.json').write_text(json.dumps(spec.raw, default=str), encoding='utf-8')
    cancel = ctl / 'cancel'
    cancel.unlink(missing_ok=True)

    proc = subprocess.Popen(  # nosec B603 - fixed argv, no shell
        [sys.executable, str(_CHILD_SCRIPT), str(ctl), spec.job_id],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    pump = threading.Thread(target=_pump_output, args=(proc,), daemon=True)
    pump.start()
    deadline = time.monotonic() + FIT_TIMEOUT_S
    try:
        while True:
            try:
                proc.wait(timeout=POLL_INTERVAL_S)
                break
            except subprocess.TimeoutExpired:
                pass
            _apply_snapshot(ctl / 'state.json', state)
            if state.cancel_requested.is_set():
                cancel.touch()
            if time.monotonic() > deadline:
                _stop_group(proc)
                msg = f'fit exceeded {FIT_TIMEOUT_S:.0f}s and was killed'
                raise FitProcessError(msg)
    finally:
        if proc.poll() is None:
            _stop_group(proc)
        pump.join(timeout=5)
        _apply_snapshot(ctl / 'state.json', state)
    if proc.returncode != 0:
        with state.lock:
            detail = state.error
        msg = f'fit process exited with code {proc.returncode}' + (f': {detail}' if detail else '')
        raise FitProcessError(msg)


def _child(ctl: Path, job_id: str) -> int:
    from trainer import run_training

    _die_with_parent()

    raw = json.loads((ctl / 'spec.json').read_text(encoding='utf-8'))
    spec = parse_and_validate_spec(job_id, raw)
    state = StatusState(job_id=job_id, campaign_id=spec.campaign_id, state='starting')
    stop = threading.Event()

    def watch() -> None:
        while not stop.wait(0.5):
            if (ctl / 'cancel').exists():
                state.cancel_requested.set()
            _write_snapshot(ctl / 'state.json', state)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        run_training(spec, state)
    finally:
        # The exception in flight, if any, ends this process non-zero with its traceback in the
        # run log; the snapshot carries its message to the parent.
        failure = sys.exc_info()[1]
        if failure is not None:
            with state.lock:
                state.state = 'failed'
                state.error = f'{type(failure).__name__}: {failure}'
        stop.set()
        watcher.join(timeout=5)
        _write_snapshot(ctl / 'state.json', state)
    return 0


if __name__ == '__main__':
    sys.exit(_child(Path(sys.argv[1]), sys.argv[2]))
