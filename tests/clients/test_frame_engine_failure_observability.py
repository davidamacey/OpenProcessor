"""A failed engine call can be traced afterwards: its gRPC code, how many attempts it took, the
process that made it and the request it served, in the log and on ``op_engine_call_failures_total``.

The 2026-11-13 diagnosis could not tell which API worker had the failing call (no process id in
the log), nor whether a failure had been retried."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from prometheus_client import REGISTRY

from src.clients import frame_engine


ROOT = Path(__file__).resolve().parents[2]


def _sample(code: str, outcome: str) -> float:
    return (
        REGISTRY.get_sample_value(
            'op_engine_call_failures_total', {'code': code, 'outcome': outcome}
        )
        or 0.0
    )


_SCRIPT = """
import asyncio, json, os, sys
from src.core import logging as op_logging
from src.clients import frame_engine
from src.domain import frame_engine_wire as wire
from src.services.curation import ingest_gpu_decode

op_logging.configure_logging(json_logs=True, log_level='INFO')

async def fails(*_a, **_k):
    raise frame_engine.EngineCallError('Stream removed', overloaded=False, code='INTERNAL', attempts=4)

frame_engine.process = fails
op_logging.request_id_ctx.set('req-abc123')
op_logging.get_logger('x').info('anything')
asyncio.run(ingest_gpu_decode.call_engine(
    object(), b'j', wire.EnginePlan(branches=(), crops='none'), caller='ingest'))
print(os.getpid(), file=sys.stderr)
"""


def _log_lines() -> tuple[list[dict[str, Any]], int]:
    """The JSON log lines of a fresh process (the real logging configuration is process-wide,
    so it is not set up inside the test session) and that process's id."""
    run = subprocess.run(  # nosec B603  fixed argv
        [sys.executable, '-c', _SCRIPT],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
        cwd=ROOT,
        env={**os.environ, 'PYTHONPATH': str(ROOT)},
    )
    lines = [json.loads(x) for x in run.stdout.splitlines() if x.startswith('{')]
    return lines, int(run.stderr.strip().splitlines()[-1])


def test_a_failed_engine_call_is_logged_with_pid_request_id_code_and_attempts() -> None:
    lines, pid = _log_lines()
    [line] = [x for x in lines if x['event'] == 'frame_engine_call_failed']
    assert line['pid'] == pid
    assert line['request_id'] == 'req-abc123'
    assert (line['code'], line['attempts']) == ('INTERNAL', 4)


def test_every_log_line_carries_the_process_id() -> None:
    lines, pid = _log_lines()
    [line] = [x for x in lines if x['event'] == 'anything']
    assert line['pid'] == pid


@pytest.mark.asyncio
async def test_retried_and_exhausted_attempts_are_counted_by_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tritonclient.utils import InferenceServerException

    from tests.clients.test_frame_engine_client import PLAN, _Client, _pool, _status_error

    async def instant(_s: float) -> None:
        return None

    monkeypatch.setattr(frame_engine, 'RETRY_SLEEP', instant)
    lost = InferenceServerException(msg='Stream removed', status='StatusCode.INTERNAL')
    retried = _sample('INTERNAL', 'retried')
    exhausted = _sample('INTERNAL', 'exhausted')
    client = _Client([lost] * 10)
    with pytest.raises(frame_engine.EngineCallError) as caught:
        await frame_engine.process(_pool(client), b'j', PLAN)
    assert caught.value.code == 'INTERNAL'
    assert caught.value.attempts == 1 + frame_engine.RETRIES
    assert _sample('INTERNAL', 'retried') == retried + frame_engine.RETRIES
    assert _sample('INTERNAL', 'exhausted') == exhausted + 1

    import grpc

    before = _sample('INVALID_ARGUMENT', 'retried'), _sample('INVALID_ARGUMENT', 'exhausted')
    with pytest.raises(frame_engine.EngineCallError):
        await frame_engine.process(
            _pool(_Client([_status_error(grpc.StatusCode.INVALID_ARGUMENT)])), b'j', PLAN
        )
    assert (_sample('INVALID_ARGUMENT', 'retried'), _sample('INVALID_ARGUMENT', 'exhausted')) == (
        before[0],
        before[1] + 1,
    )
