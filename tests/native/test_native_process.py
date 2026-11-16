"""The engine as a process: how it starts, refuses a bad start, drains on SIGTERM, exposes its
metrics and keeps an idle channel (marker ``native``, no GPU; every wait is bounded and the
process is stopped by the exact pid this test started)."""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import grpc
import httpx
import pytest
from tritonclient.grpc import aio, service_pb2, service_pb2_grpc
from tritonclient.utils import InferenceServerException

from src.clients import frame_engine as client

from .engines import FRAME, PHOTO, RustEngine, free_ports, inputs, script


if TYPE_CHECKING:
    from .conftest import StartEngine

pytestmark = pytest.mark.native
async_test = pytest.mark.asyncio(loop_scope='function')

PROTOCOL = json.loads(
    (Path(__file__).resolve().parents[2] / 'contracts' / 'json' / 'frame_engine_v1.json').read_text(
        encoding='utf-8'
    )
)


def run_once(
    binary: Path, args: list[str], env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # the release binary of this checkout, a fixed argument list
        [str(binary), *args], env={'PATH': '/usr/bin:/bin', **env}, capture_output=True, text=True,
        timeout=30, check=False,
    )  # fmt: skip


def test_the_binary_wants_a_pipeline(binary: Path) -> None:
    done = run_once(binary, [], {})
    assert done.returncode == 2
    assert '--pipeline' in done.stderr
    done = run_once(binary, ['--help'], {})
    assert done.returncode == 0
    assert 'op-frame-engine' in done.stdout
    assert 'stub' in done.stdout


@pytest.mark.parametrize(
    ('name', 'value', 'text'),
    [
        ('OP_FRAME_ENGINE_SWITCH_INTERVAL_MS', '0.1', 'means nothing to the native engine'),
        ('OP_FRAME_ENGINE_WORKERS', '0', "OP_FRAME_ENGINE_WORKERS='0': outside [1, 256]"),
        ('OP_FRAME_ENGINE_WORKERS', 'many', 'not a number'),
        ('OP_FRAME_ENGINE_TRANSPORT', 'system', 'one of cuda, body'),
        ('OP_FRAME_ENGINE_BUDGET_MIB', '8192', 'exceeds OP_FRAME_ENGINE_POOL_MIB'),
    ],
)
def test_a_bad_setting_refuses_the_start_with_exit_1_naming_the_variable(
    binary: Path, name: str, value: str, text: str
) -> None:
    done = run_once(binary, ['--pipeline', 'stub'], {name: value})
    assert done.returncode == 1, done.stdout + done.stderr
    assert name in done.stdout + done.stderr
    assert text in done.stdout + done.stderr


def test_a_port_that_is_taken_refuses_the_start(binary: Path) -> None:
    import socket

    with socket.socket() as taken:
        taken.bind(('0.0.0.0', 0))
        taken.listen()
        port = taken.getsockname()[1]
        done = run_once(binary, ['--pipeline', 'stub'], {
            'OP_FRAME_ENGINE_PORT': str(port), 'OP_FRAME_ENGINE_METRICS_PORT': str(free_ports(1)[0]),
        })  # fmt: skip
    assert done.returncode == 1
    assert 'in use' in done.stdout + done.stderr


@async_test
async def test_the_health_subcommand_follows_model_ready(start: StartEngine, binary: Path) -> None:
    """The container's health check: exit 0 while the engine answers ModelReady, 1 when it does
    not, 1 when no engine listens."""
    engine = await start('rust')
    assert isinstance(engine, RustEngine)
    port = engine.target.rsplit(':', 1)[1]
    env = {'OP_FRAME_ENGINE_PORT': port}
    done = await asyncio.to_thread(run_once, binary, ['health'], env)
    assert done.returncode == 0, done.stdout + done.stderr
    await engine.set_ready(False)
    await engine.wait_model_ready(False)
    done = await asyncio.to_thread(run_once, binary, ['health'], env)
    assert done.returncode == 1
    assert 'not ready' in done.stdout + done.stderr
    nobody = await asyncio.to_thread(
        run_once, binary, ['health', '--timeout-seconds', '0.5'],
        {'OP_FRAME_ENGINE_PORT': str(free_ports(1)[0])},
    )  # fmt: skip
    assert nobody.returncode == 1
    assert 'no engine' in nobody.stdout + nobody.stderr


@async_test
async def test_sigterm_stops_an_idle_engine_with_exit_0_at_once(start: StartEngine) -> None:
    engine = await start('rust')
    assert isinstance(engine, RustEngine)
    started = time.monotonic()
    await engine.begin_drain()
    assert await asyncio.to_thread(engine.proc.wait, 10) == 0
    assert time.monotonic() - started < 3.0
    assert 'stopped' in engine.output()


@async_test
async def test_sigterm_lets_a_photo_in_flight_finish_and_then_exits_0(start: StartEngine) -> None:
    engine = await start('rust', request_deadline=10.0, drain_seconds=10.0)
    assert isinstance(engine, RustEngine)
    t = aio.InferenceServerClient(engine.target)
    try:
        held = asyncio.create_task(t.infer('frame_engine', inputs(script(hold_ms=1200), FRAME)))
        await asyncio.sleep(0.4)
        await engine.begin_drain()
        assert (await held).as_numpy('status')[0] == b'ok'
        assert await asyncio.to_thread(engine.proc.wait, 10) == 0
    finally:
        await t.close()


@async_test
async def test_a_drain_that_runs_out_ends_the_process_anyway(start: StartEngine) -> None:
    engine = await start('rust', request_deadline=30.0, drain_seconds=0.5)
    assert isinstance(engine, RustEngine)
    t = aio.InferenceServerClient(engine.target)
    try:
        held = asyncio.create_task(t.infer('frame_engine', inputs(script(hold_ms=20000), FRAME)))
        await asyncio.sleep(0.4)
        started = time.monotonic()
        await engine.begin_drain()
        assert await asyncio.to_thread(engine.proc.wait, 10) == 0
        assert time.monotonic() - started < 3.0
        with pytest.raises(InferenceServerException):
            await held
    finally:
        await t.close()


@async_test
async def test_the_metrics_endpoint_has_the_python_engines_names(start: StartEngine) -> None:
    engine = await start('rust', max_in_flight=1, request_deadline=10.0)
    assert isinstance(engine, RustEngine)
    t = aio.InferenceServerClient(engine.target)
    try:
        held = asyncio.create_task(t.infer('frame_engine', inputs(script(hold_ms=800), FRAME)))
        for _ in range(100):
            if await engine.in_flight() == 1:
                break
            await asyncio.sleep(0.02)
        with pytest.raises(InferenceServerException):
            await t.infer('frame_engine', inputs(PHOTO, FRAME))
        await held
    finally:
        await t.close()
    async with httpx.AsyncClient(timeout=5.0) as http:
        reply = await http.get(engine.metrics_url)
    assert reply.headers['content-type'].startswith('application/openmetrics-text')
    text = reply.text
    assert text.endswith('# EOF\n')
    for family in PROTOCOL['metrics']:
        if not family['labels']:  # the labelled ones appear with their first sample
            assert f'# TYPE {family["name"]} {family["type"]}\n' in text, family['name']
    assert 'op_frame_engine_overloads_total{cause="in_flight"} 1' in text
    assert 'op_frame_engine_in_flight 0' in text
    assert 'process_resident_memory_bytes ' in text


@pytest.mark.slow
@async_test
async def test_an_idle_channel_pinged_for_65_seconds_is_never_sent_away(start: StartEngine) -> None:
    """The API's channels ping every 30 s whether or not a call is in flight
    (``ENGINE_CHANNEL_OPTIONS``); a server that answered with GOAWAY ``too_many_pings`` would
    close the channel and the state would leave READY. tonic enforces no ping policy (h2
    answers every ping), which this proves on the wire with the production options."""
    engine = await start('rust')
    assert isinstance(engine, RustEngine)
    channel = grpc.aio.insecure_channel(engine.target, options=list(client.ENGINE_CHANNEL_OPTIONS))
    try:
        live = service_pb2_grpc.GRPCInferenceServiceStub(channel)
        assert (await live.ServerLive(service_pb2.ServerLiveRequest(), timeout=5.0)).live
        states: list[Any] = []
        deadline = time.monotonic() + 65.0
        while time.monotonic() < deadline:
            states.append(channel.get_state(try_to_connect=False))
            await asyncio.sleep(0.5)
        assert set(states) == {grpc.ChannelConnectivity.READY}, (
            f'the channel left READY: {set(states)}'
        )
        assert (await live.ServerLive(service_pb2.ServerLiveRequest(), timeout=5.0)).live
    finally:
        await channel.close()
    assert 'GOAWAY' not in engine.output()
