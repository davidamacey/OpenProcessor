"""The ``gpu_export`` consumer: the export image's Celery app and the ``model_export`` handler.

    celery -A export.export_tasks worker -Q gpu_export -c 1 --max-tasks-per-child=1

``POST /models/upload`` and ``POST /models/{name}/export`` start a ``model_export`` job
(``src/services/model_export_jobs.py``). This handler runs the export script as a subprocess,
records each engine it builds in ``model_engines`` with its contracts and measured cost
(``export/engine_facts.py``), and loads the engines into Triton when asked. The phases are the
job's ``progress.stage`` (``validating``, ``exporting``, ``loading``); a cancel is honoured between
phases and while the build prints. The kind's time limit (4 h) replaces the old watchdog: when the
wrapper cancels the handler at the soft limit, the subprocess is killed here.

Celery: https://docs.celeryq.dev/en/stable/userguide/workers.html (``-c 1`` with
``--max-tasks-per-child=1`` gives every build a fresh process that releases its GPU memory).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
import sys
from collections import deque
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import httpx
from celery.signals import worker_init

from src.config.retired_env import reject_retired_env
from src.config.settings import get_settings
from src.db.repos import engine_builds
from src.services.triton_control import TritonControlService
from src.tasks.runtime.gpu_app import make_app
from src.tasks.runtime.log_context import install_log_context
from src.tasks.runtime.register import register_job


if TYPE_CHECKING:
    from collections.abc import Callable

    from src.tasks.runtime.context import JobContext

# The export scripts import their siblings by bare name (they run as scripts), so this module
# does the same instead of a second, package-qualified module object for each of them.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine_facts
import trt_build_settings
from export_worker import APP_DIR, build_command, engines_to_load


logger = logging.getLogger('export_tasks')

app = make_app(include=['export.export_tasks'])
install_log_context()

MODELS_DIR = Path('/app/models')
READ_LIMIT = 16 * 1024 * 1024  # a build prints progress bars as one very long "line"
POLL_S = 2.0
TAIL_LINES = 20
METRICS_PORT = 8002
_PROGRESS = {'validating': 5, 'exporting': 20, 'loading': 90}


@worker_init.connect(weak=False)
def _on_worker_init(**_: Any) -> None:
    """Refuse to serve on a retired setting, an unreachable database or another schema revision.

    Celery logs an ``Exception`` raised by a signal receiver and carries on, so the refusal is
    a ``SystemExit`` (https://docs.celeryq.dev/en/stable/userguide/signals.html).
    """
    from src.db import schema_check

    reject_retired_env()
    schema_check.require_current_schema()


def _phase_of(line: str) -> tuple[float, str] | None:
    """(progress percent, step text) for an export script output line that marks a step."""
    if 'ONNX' in line and 'Export' in line:
        return 40.0, 'Exporting ONNX End2End'
    if 'TensorRT' in line or 'Building' in line:
        return 60.0, 'Building TensorRT engine (this takes 2-5 min)'
    if 'Engine saved' in line:
        return 85.0, 'Finalizing export'
    return None


async def _stage(ctx: JobContext, stage: str, step: str, **counters: Any) -> None:
    await ctx.progress(_PROGRESS[stage], 100, stage, {'step': step, **counters}, force=True)


async def _run_build(ctx: JobContext, cmd: list[str], step_times: dict[str, float]) -> bool:
    """Run the export script; False when the job was cancelled and the script was stopped."""
    tail: deque[str] = deque(maxlen=TAIL_LINES)
    phase, phase_start = 'initialization', perf_counter()
    logger.info('[%s] running: %s', ctx.job_id, ' '.join(cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=APP_DIR,
        env={**os.environ, 'PYTHONPATH': str(APP_DIR)},
        limit=READ_LIMIT,
    )
    assert proc.stdout is not None  # PIPE was requested above
    try:
        while True:
            try:
                raw = await asyncio.wait_for(proc.stdout.readline(), POLL_S)
            except TimeoutError:
                if ctx.cancelled():
                    return False
                continue
            if not raw:
                break
            line = raw.decode(errors='replace').strip()
            tail.append(line)
            hit = _phase_of(line)
            if hit is not None and hit[1] != phase:
                step_times[phase] = round(perf_counter() - phase_start, 2)
                phase, phase_start = hit[1], perf_counter()
                await ctx.progress(
                    int(hit[0]), 100, 'exporting', {'step': hit[1], 'step_times': step_times}
                )
            if ctx.cancelled():
                return False
        returncode = await proc.wait()
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
    step_times[phase] = round(perf_counter() - phase_start, 2)
    if returncode != 0:
        raise RuntimeError(f'Export failed (exit {returncode}):\n' + '\n'.join(tail))
    return True


async def _triton_memory_mib(host: str) -> float:
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f'http://{host}:{METRICS_PORT}/metrics')
        response.raise_for_status()
    return engine_facts.total_gpu_memory_mib(response.text)


async def _load_and_measure(name: str, *, keep: bool) -> tuple[bool, int | None]:
    """Load ``name`` into Triton once and read what it cost; unload again unless ``keep``."""
    triton = TritonControlService()
    host = get_settings().triton_host
    try:
        before = await _triton_memory_mib(host)
    except httpx.HTTPError:
        logger.warning('no Triton metrics at %s: %s is loaded without a cost measure', host, name)
        before = None
    ok, message = await triton.load_model(name)
    if not ok:
        logger.warning('could not load %s into Triton: %s', name, message)
        return False, None
    measured = None
    if before is not None:
        with contextlib.suppress(httpx.HTTPError):
            measured = engine_facts.measured_mib(before, await _triton_memory_mib(host))
    if not keep:
        await triton.unload_model(name)
    return True, measured


def _plan_path(engine_name: str) -> Path:
    return MODELS_DIR / engine_name / '1' / 'model.plan'


def _onnx_sha256(engine_name: str) -> str | None:
    """Hash of the ONNX the engine came from, when the export left one next to it."""
    base = engine_name.removesuffix('_trt_end2end').removesuffix('_trt')
    for candidate in (
        MODELS_DIR / f'{base}_end2end' / '1' / 'model.onnx',
        APP_DIR / 'export' / f'{base}.onnx',
    ):
        if candidate.is_file():
            return engine_facts.file_sha256(candidate)
    return None


async def _begin_engines(ctx: JobContext, names: list[str], max_batch: int) -> dict[str, Any]:
    """One ``building`` row per engine; returns ``name -> row id``."""
    base = str(ctx.params['triton_name'])

    def write(sync: Any) -> dict[str, Any]:
        model = engine_builds.latest_model_version(sync, base)
        if model is None:
            raise LookupError(f'no models row for {base}: start the export through the API')
        return {
            name: engine_builds.begin_engine(
                sync,
                model_id=model.id,
                triton_model_name=name,
                max_batch=max_batch,
                build_job_id=ctx.job_id,
            )
            for name in names
        }

    async with ctx.session() as session:
        return await session.run_sync(write)


async def _fail_building(ctx: JobContext) -> None:
    async with ctx.session() as session:
        await session.run_sync(lambda sync: engine_builds.fail_building(sync, ctx.job_id))


async def _facts_of(name: str, head_kind: str, max_batch: int) -> engine_builds.EngineFacts:
    """Read the built plan, check it against the declared contract and describe the build."""
    plan_path = _plan_path(name)
    if not plan_path.is_file():
        raise FileNotFoundError(f'the export did not leave {plan_path}')
    plan = await asyncio.to_thread(engine_facts.read_plan, plan_path)
    input_contract, output_contract, size = engine_facts.contracts_for(
        plan, engine_name=name, head_kind=head_kind
    )
    opt = max(1, max_batch // 2)
    profile = trt_build_settings.build_target()
    flags = f'target={profile} workspace={trt_build_settings.workspace_gb(name)}G'
    return engine_builds.EngineFacts(
        input_size=size,
        input_contract=input_contract,
        output_contract=output_contract,
        build_profile=profile,
        build_settings={
            'workspace_gb': trt_build_settings.workspace_gb(name),
            'max_batch': max_batch,
            'opt_batch': opt,
            'precision': engine_facts.PRECISION,
        },
        build_stamp=engine_facts.build_stamp(max_batch, opt, size, flags, 'uint8'),
        onnx_sha256=await asyncio.to_thread(_onnx_sha256, name),
        engine_sha256=await asyncio.to_thread(engine_facts.file_sha256, plan_path),
        device_mib_measured=None,
        max_batch=max_batch,
        opt_batch=opt,
        precision=engine_facts.PRECISION,
        trt_version=plan.trt_version,
        gpu_arch=plan.gpu_arch,
        engine_storage_key=f'models/{name}/1/model.plan',
    )


def _remove_partial(names: list[str], had_ready: set[str]) -> None:
    """A cancelled build leaves no half-written engine: remove the directories this job made.

    A name that already had a ready engine keeps its directory (the previous build is all that
    is left of it)."""
    for name in names:
        if name not in had_ready:
            shutil.rmtree(MODELS_DIR / name, ignore_errors=True)


async def _had_ready(ctx: JobContext, names: list[str]) -> set[str]:
    def read(sync: Any) -> set[str]:
        ready = engine_builds.ready_names(sync)
        return {n for n in names if n in ready}

    async with ctx.session() as session:
        return await session.run_sync(read)


async def _load_all(
    names: list[str], facts: dict[str, engine_builds.EngineFacts], auto_load: bool
) -> bool:
    """Load each engine once to measure it; True when ``auto_load`` and at least one loaded."""
    loaded = False
    for name in names:
        ok, mib = await _load_and_measure(name, keep=auto_load)
        loaded = loaded or (ok and auto_load)
        facts[name] = replace(facts[name], device_mib_measured=mib)
    return loaded


def _finisher(row: Any, fact: engine_builds.EngineFacts) -> Callable[[Any], None]:
    def finish(sync: Any) -> None:
        engine_builds.finish_engine(sync, row, fact)

    return finish


async def export_model(ctx: JobContext) -> dict[str, Any]:
    """Run one export to ``completed``, ``failed`` (an exception) or ``cancelled``."""
    job = dict(ctx.params)
    started = perf_counter()
    step_times: dict[str, float] = {}
    await _stage(ctx, 'validating', 'Validating the export request')
    names = engines_to_load(job)  # raises when the job has no head kind: fail closed
    cmd = build_command(job)
    if not Path(job['pt_file']).is_file():
        raise FileNotFoundError(f'weights not found: {job["pt_file"]}')
    max_batch = int(job['max_batch'])
    had_ready = await _had_ready(ctx, names)
    ids = await _begin_engines(ctx, names, max_batch)
    try:
        if ctx.cancelled():
            return await _cancelled(ctx, names, had_ready)
        await _stage(ctx, 'exporting', f'Exporting: {", ".join(job["formats"])}')
        if not await _run_build(ctx, cmd, step_times) or ctx.cancelled():
            return await _cancelled(ctx, names, had_ready)
        facts = {name: await _facts_of(name, job['head_kind'], max_batch) for name in names}
        await _stage(ctx, 'loading', 'Loading into Triton', step_times=step_times)
        t0 = perf_counter()
        loaded = await _load_all(names, facts, bool(job.get('auto_load')))
        step_times['triton_load'] = round(perf_counter() - t0, 2)
        async with ctx.session() as session:
            for name, fact in facts.items():
                await session.run_sync(_finisher(ids[name], fact))
    except BaseException:
        await asyncio.shield(_fail_building(ctx))
        raise
    total = round(perf_counter() - started, 2)
    logger.info('[%s] export completed in %.1fs; step times: %s', ctx.job_id, total, step_times)
    return {
        'triton_loaded': loaded,
        'export_duration_seconds': total,
        'step_times': step_times,
        'engines': names,
    }


async def _cancelled(ctx: JobContext, names: list[str], had_ready: set[str]) -> dict[str, Any]:
    await _fail_building(ctx)
    await asyncio.to_thread(_remove_partial, names, had_ready)
    return {'cancelled': True}


register_job('model_export', export_model)
