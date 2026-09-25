"""GPU arbiter — coordinate GPU-resident services around training runs.

Two regimes:

* **Single-GPU training**: the trainer takes one configured GPU-resident
  service; a paired worker (if configured) is paused via a sentinel file
  so it doesn't fight the trainer for CPU/RAM, but its model server stays
  loaded — no cold-start penalty when training finishes.
* **Multi-GPU training**: the trainer claims every configured GPU-resident
  container. We **stop those containers entirely** for the run and restart
  them afterward, so all configured GPUs are fully free for training.

The API claims GPUs on job start (``claim_gpus_for_training``), which
writes a **training lock** and stops the configured containers. The API's
reconcile loop (``reconcile_on_startup``) then *enforces* that state on
every tick: re-stops anything that comes back up while a run is active,
and once nothing is active clears the lock and restarts services (also
crash recovery). Not self-starting -- wired into the FastAPI lifespan in
:mod:`src.main` (startup reconcile + a task ticking every
``ARBITER_RECONCILE_INTERVAL_SECONDS``); an embedding host needs the
equivalent (see ``tests/integration/test_gpu_arbiter_lifespan.py``).

**Deployment facts vs. mechanism.** Which GPU ids a job may target, which
containers to stop/start, and which container is "the trainer" are
deployment facts (:class:`src.config.gpu_arbiter.GpuArbiterConfig`).
Every public function takes them as config-derived-default parameters,
so an unconfigured install degrades to a no-op
(``tests/curation/test_gpu_arbiter_config.py``).

Implementation note: container control uses the docker SDK over the mounted host socket
(``/var/run/docker.sock``, opt-in via ``docker-compose.gpu-arbiter.yml`` -- see G-15), by name, so
restart preserves each container's config (GPU pins included). A claim that must stop a configured
container and can't (SDK/socket unavailable, or the stop fails) raises
:class:`GpuArbiterStopFailedError` rather than falling back to a sentinel-only pause, which pauses
a paired worker but leaves a sibling container running on the shared GPU. Release/resume keeps a
logged best-effort fallback.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import get_curation_config, get_gpu_arbiter_config
from src.core.logging import get_logger


logger = get_logger(__name__)
_docker_unavailable_logged = [False]  # G-15: warn once per outage, not per call


# =============================================================================
# Constants
# =============================================================================


def _state_dir() -> Path:
    return get_curation_config().state_dir


def _default_sentinel_path() -> Path:
    # Suffix must match CurationConfig.pause_sentinel_path (readers).
    return _state_dir() / 'vlm_worker' / 'pause.sentinel'


def _default_lock_path() -> Path:
    return _state_dir() / 'training_gpus.lock'


HEALTH_WAIT_SECONDS = 90
_DOCKER_STOP_TIMEOUT = 30

# A run owns the GPUs until its status.json reaches one of these terminal
# states. MUST stay in sync with the trainer's own terminal-state set.
# Reconcile treats *any* non-terminal state (queued / starting / running /
# exporting -- and any future or unrecognized active state) as "run live,
# keep GPUs reserved". Using the terminal set as the source of truth --
# rather than an allowlist of active states -- is deliberately fail-safe: a
# multi-day run that sits in ``running`` for hours, or a trainer that
# introduces a new active state, can never be misread as "idle" and have
# the configured GPU-resident containers restarted underneath it.
TRAINER_TERMINAL_STATES = frozenset({'finished', 'failed', 'cancelled', 'skipped'})

# Training lock -- written the instant we claim GPUs (BEFORE we stop the
# containers and BEFORE the trainer's job.json/status.json appear). Its only
# job is to close the brief **claim->write race**: the reconcile loop can run
# in any/all uvicorn workers, and in the sub-second window before job.json is
# visible it would otherwise see "no active job" and restart the very
# services we just stopped. Once job.json exists the ``/jobs`` scan is
# authoritative for the whole run; the lock is therefore only honored for
# ``LOCK_GRACE_SECONDS`` (a short race-closer window, NOT the run duration)
# and ages out + is cleared after that.
LOCK_GRACE_SECONDS = 90.0


def bakeoff_active(*, jobs_dir: Path | None = None) -> bool:
    """True if a bake-off job is queued or running (job.json still present).

    ``jobs_dir`` defaults to ``GpuArbiterConfig.bakeoff_jobs_dir`` (always
    set: ``OP_BAKEOFF_JOBS_DIR`` or ``<state_dir>/bakeoff_jobs``, the same
    dir the bake-off router writes into). A missing dir means nothing queued.
    """
    configured = jobs_dir if jobs_dir is not None else get_gpu_arbiter_config().bakeoff_jobs_dir
    if not configured:
        return False
    target = Path(configured)
    try:
        return any(target.glob('*.job.json'))
    except OSError:
        return False


# =============================================================================
# Result + state types
# =============================================================================


@dataclass(frozen=True)
class ArbiterAction:
    """Returned by every arbiter call so the caller can log what happened."""

    action: str  # 'sentinel_set' | 'sentinel_cleared' | 'gpu_services_stopped' | 'gpu_services_started' | 'noop'
    detail: str = ''


class GpuArbiterStopFailedError(RuntimeError):
    """A claim needed to stop a configured GPU-resident container and
    couldn't (docker SDK/socket unavailable, or the stop call failed).
    Used to fall back to a sentinel-only pause instead, which doesn't
    stop the container -- training could start next to it on the same
    GPU. Callers (``/train/start``, ``/train/start_campaign``) must
    catch this and refuse with 409 rather than proceed.
    """


# =============================================================================
# Public API
# =============================================================================


def parse_cuda_visible_devices(spec_value: str | None) -> list[int]:
    """Parse a CUDA_VISIBLE_DEVICES string into the integer device list.

    >>> parse_cuda_visible_devices('0,2')
    [0, 2]
    >>> parse_cuda_visible_devices(None)
    []
    >>> parse_cuda_visible_devices('  0 ,  1 ')
    [0, 1]
    """
    if not spec_value:
        return []
    out: list[int] = []
    for raw in spec_value.split(','):
        tok = raw.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError:
            logger.warning('cuda_visible_devices_unparseable', token=tok)
    return out


def needs_multi_gpu_stop(cuda_visible_devices: str | None) -> bool:
    """Does this claim span more than one GPU id?

    Only used to decide the fallback behavior for *unscoped* containers
    (see :func:`containers_to_stop`) -- an unscoped container is stopped
    only when the claim spans more than one GPU, matching the original
    (pre-scoping) semantics. Whether any container actually gets stopped
    is decided by :func:`containers_to_stop` / :func:`needs_service_stop`,
    not this function directly.
    """
    devices = parse_cuda_visible_devices(cuda_visible_devices)
    return len(devices) > 1


def containers_to_stop(cuda_visible_devices: str | None) -> tuple[str, ...]:
    """Ordered names of configured containers this claim must stop.

    A *scoped* container (``name@ids`` in ``OP_GPU_ARBITER_CONTAINERS``,
    see :class:`src.config.gpu_arbiter.GpuArbiterConfig`) is stopped
    whenever the claim intersects its GPU set, regardless of claim size --
    a lone-GPU claim on a GPU that hosts a large service must stop that
    service, not just pause a paired worker. An *unscoped* container keeps
    the original behavior: stopped only when the claim spans more than one
    GPU. Order follows ``GpuArbiterConfig.container_gpus`` (== the order
    containers were configured in).
    """
    claim = frozenset(parse_cuda_visible_devices(cuda_visible_devices))
    multi_gpu_claim = len(claim) > 1
    names: list[str] = []
    for name, scope in get_gpu_arbiter_config().container_gpus:
        if scope is None:
            if multi_gpu_claim:
                names.append(name)
        elif claim & scope:
            names.append(name)
    return tuple(names)


def needs_service_stop(cuda_visible_devices: str | None) -> bool:
    """``True`` iff this claim requires stopping at least one configured container."""
    return bool(containers_to_stop(cuda_visible_devices))


def sentinel_path(custom: Path | None = None) -> Path:
    return custom or _default_sentinel_path()


# ---- training lock (claim->write race-closer + enforce source of truth) ----


def lock_path(custom: Path | None = None) -> Path:
    return custom or _default_lock_path()


def set_training_lock(
    cuda_visible_devices: str | None,
    *,
    path: Path | None = None,
) -> None:
    """Record that a training run owns the configured GPU-resident services."""
    target = lock_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({'claimed_at': time.time(), 'cuda_visible_devices': cuda_visible_devices or ''}),
        encoding='utf-8',
    )
    logger.info('arbiter_training_lock_set', path=str(target), devices=cuda_visible_devices)


def clear_training_lock(*, path: Path | None = None) -> None:
    """Remove the training lock. Idempotent."""
    target = lock_path(path)
    try:
        target.unlink()
    except FileNotFoundError:
        return
    except OSError as exc:
        logger.warning('arbiter_training_lock_unlink_failed', error=str(exc))
        return
    logger.info('arbiter_training_lock_cleared', path=str(target))


def read_training_lock(*, path: Path | None = None) -> dict[str, Any] | None:
    """Return the lock payload, or ``None`` if absent/unreadable."""
    target = lock_path(path)
    try:
        return json.loads(target.read_text(encoding='utf-8'))
    except (FileNotFoundError, ValueError, OSError):
        return None


# ---- pause / resume (single-GPU path) ----------------------------------


async def pause_gpu_worker(
    *,
    sentinel: Path | None = None,
) -> ArbiterAction:
    """Touch the pause sentinel so a paired CPU-side worker idles.

    The GPU-resident model server(s) are left alone -- only the polling
    worker backs off. This is the single-GPU regime: trainer and the
    other configured service share the host's CPU/RAM but live on
    different GPUs.
    """
    target = sentinel_path(sentinel)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return ArbiterAction(action='noop', detail=f'sentinel already at {target}')
    target.touch()
    logger.info('arbiter_sentinel_set', path=str(target))
    return ArbiterAction(action='sentinel_set', detail=str(target))


async def resume_gpu_worker(
    *,
    sentinel: Path | None = None,
) -> ArbiterAction:
    """Remove the pause sentinel. Idempotent."""
    target = sentinel_path(sentinel)
    if not target.exists():
        return ArbiterAction(action='noop', detail=f'no sentinel at {target}')
    try:
        target.unlink()
    except OSError as exc:
        logger.warning('arbiter_sentinel_unlink_failed', error=str(exc))
        return ArbiterAction(action='noop', detail=f'unlink failed: {exc}')
    logger.info('arbiter_sentinel_cleared', path=str(target))
    return ArbiterAction(action='sentinel_cleared', detail=str(target))


# ---- stop / start (multi-GPU path) --------------------------------------


def _docker_client() -> Any:
    """Return a docker SDK client over the mounted socket, or ``None`` (warns once
    per outage, not per call -- G-15). Imported lazily so the module stays
    importable without the ``docker`` package or socket (unit tests)."""
    try:
        import docker  # type: ignore[import-untyped]

        client = docker.from_env()  # type: ignore[attr-defined]
        client.ping()
        _docker_unavailable_logged[0] = False
        return client
    except Exception as exc:
        if not _docker_unavailable_logged[0]:  # G-15: warn once per outage
            logger.warning('arbiter_docker_unavailable', error=str(exc))
            _docker_unavailable_logged[0] = True
        return None


def docker_client_available() -> bool:
    """Preflight probe -- can a docker client reach the daemon?"""
    return _docker_client() is not None


def _stop_containers_sync(client: Any, names: tuple[str, ...]) -> list[str]:
    """Stop each named container if running. Returns the names stopped."""
    import docker.errors

    stopped: list[str] = []
    for name in names:
        try:
            container = client.containers.get(name)
        except docker.errors.NotFound:
            continue
        if container.status == 'running':
            container.stop(timeout=_DOCKER_STOP_TIMEOUT)
            stopped.append(name)
    return stopped


def _start_containers_sync(client: Any, names: tuple[str, ...]) -> list[str]:
    """Start each named container if not already running. Returns names started."""
    import docker.errors

    started: list[str] = []
    for name in names:
        try:
            container = client.containers.get(name)
        except docker.errors.NotFound:
            continue
        container.reload()
        if container.status != 'running':
            container.start()
            started.append(name)
    return started


async def stop_gpu_services(
    *,
    containers: tuple[str, ...] | None = None,
) -> ArbiterAction:
    """Stop the configured GPU-resident containers to free every configured GPU.

    ``containers`` defaults to ``GpuArbiterConfig.containers``. Empty
    (generic-install default) = pure no-op, no docker SDK touched.

    Otherwise uses the docker SDK over the mounted socket; restart later
    preserves each container's original config (GPU pins included).

    Fails closed -- raises :class:`GpuArbiterStopFailedError` if the
    docker SDK/socket is unavailable or the stop call fails, rather than
    falling back to a sentinel (which pauses a paired *worker*, not a
    sibling container sharing the GPU). Callers must not proceed (or
    write job.json) when this raises.
    """
    names = containers if containers is not None else get_gpu_arbiter_config().containers
    if not names:
        return ArbiterAction(action='noop', detail='no GPU-resident containers configured')
    client = _docker_client()
    if client is None:
        raise GpuArbiterStopFailedError(
            f'docker SDK/socket unavailable in the API container -- cannot stop {names!r}'
        )
    try:
        stopped = await asyncio.to_thread(_stop_containers_sync, client, names)
    except Exception as exc:
        logger.warning('arbiter_gpu_stop_failed', error=str(exc))
        raise GpuArbiterStopFailedError(f'stop failed for {names!r}: {exc}') from exc
    # Belt-and-suspenders: also set the sentinel so a worker that somehow
    # comes back up mid-run still pauses.
    await pause_gpu_worker()
    logger.info('arbiter_gpu_services_stopped', stopped=stopped)
    return ArbiterAction(
        action='gpu_services_stopped',
        detail=','.join(stopped) if stopped else 'none were running',
    )


async def start_gpu_services(
    *,
    containers: tuple[str, ...] | None = None,
) -> ArbiterAction:
    """Restart the configured GPU-resident containers and clear the pause sentinel."""
    names = containers if containers is not None else get_gpu_arbiter_config().containers
    if not names:
        return ArbiterAction(action='noop', detail='no GPU-resident containers configured')
    client = _docker_client()
    if client is None:
        await resume_gpu_worker()
        return ArbiterAction(
            action='sentinel_cleared',
            detail='docker SDK unavailable; cleared sentinel only',
        )
    try:
        started = await asyncio.to_thread(_start_containers_sync, client, names)
    except Exception as exc:
        logger.warning('arbiter_gpu_start_failed', error=str(exc))
        return ArbiterAction(action='noop', detail=f'start failed: {exc}')
    await resume_gpu_worker()
    logger.info('arbiter_gpu_services_started', started=started)
    return ArbiterAction(
        action='gpu_services_started',
        detail=','.join(started) if started else 'all already running',
    )


# ---- trainer reachability (preflight) ---------------------------------
#
# The probe itself lives in trainer_reachability.py (kept out of this file
# to stay under the 700-LOC ratchet) -- it borrows _container_status_sync/
# _docker_client below for its optional docker-confirmation path. Imported
# and re-exported at the bottom of this module so existing callers
# (src.routers.curation_train, this module's own __all__) don't need to
# know it moved.


def _container_status_sync(client: Any, name: str) -> str | None:
    """Return the named container's status, or ``None`` if it doesn't exist."""
    import docker.errors

    try:
        container = client.containers.get(name)
    except docker.errors.NotFound:
        return None
    container.reload()
    return container.status


# ---- top-level dispatch ------------------------------------------------


async def claim_gpus_for_training(
    cuda_visible_devices: str | None,
) -> ArbiterAction:
    """Apply the right pause/stop strategy before starting a training run.

    Called at the moment a job transitions queued -> starting. Returns
    the action taken so the caller can log it into ``status.json``.

    The training lock is written *first* -- before we stop the
    containers and before the trainer writes ``job.json`` -- so the
    reconcile loop can never race in during that window and restart the
    services we are about to stop.

    Which containers stop is decided by GPU scope
    (:func:`containers_to_stop`), not claim size: a single-GPU claim that
    intersects a *scoped* container's GPU set stops that container just
    like a multi-GPU claim would. ``stop_gpu_services`` also sets the
    pause sentinel (belt-and-suspenders). If it raises
    :class:`GpuArbiterStopFailedError`, the lock written above is
    cleared before the exception propagates -- a refused claim never
    leaves a stale lock behind.
    """
    set_training_lock(cuda_visible_devices)
    names = containers_to_stop(cuda_visible_devices)
    if not names:
        return await pause_gpu_worker()
    try:
        return await stop_gpu_services(containers=names)
    except GpuArbiterStopFailedError:
        clear_training_lock()
        raise


async def release_gpus_after_training(
    cuda_visible_devices: str | None,
) -> ArbiterAction:
    """Inverse of :func:`claim_gpus_for_training`.

    Restarts exactly the containers :func:`containers_to_stop` says this
    claim stopped; otherwise clears the pause sentinel. Idempotent --
    safe when nothing was stopped (e.g. crash before claim). The API's
    reconcile loop (:func:`reconcile_on_startup`) is the real backstop
    since the trainer container may have no docker socket.
    """
    clear_training_lock()
    names = containers_to_stop(cuda_visible_devices)
    if names:
        return await start_gpu_services(containers=names)
    return await resume_gpu_worker()


# ---- recovery on API startup -------------------------------------------


def _resolve_train_jobs_dir() -> Path:
    """The trainer's jobs directory, honoring ``OP_TRAIN_JOBS_DIR``.

    Delegates to the training-jobs module rather than re-reading the env
    var so the arbiter can never scan a different directory than the one
    jobs are actually written to. Imported lazily to keep this module
    importable on its own.
    """
    from src.services.training.jobs import _resolve_jobs_dir

    return _resolve_jobs_dir()


async def reconcile_on_startup(
    *,
    train_jobs_dir: Path | None = None,
    sentinel: Path | None = None,
) -> ArbiterAction:
    """Enforce the desired GPU-service state -- both directions.

    ``train_jobs_dir`` defaults to the deployment's configured jobs
    directory (:func:`_resolve_train_jobs_dir`); callers pass it
    explicitly only to point at a test fixture.

    Runs once at API startup *and* on a periodic loop (wired in
    :mod:`src.main`'s lifespan), in every uvicorn worker. Single
    authority for whether the configured GPU-resident containers should
    be down (a run owns the GPUs) or up (nothing active), and drives
    them toward that state every tick -- idempotent, so running in every
    worker is safe re-enforcement, not a race.

    A run owns the GPUs when either: a ``*.job.json`` exists whose
    ``*.status.json`` hasn't reached a :data:`TRAINER_TERMINAL_STATES`
    state (no status yet = just-claimed; unknown/non-terminal = still
    live); or the training lock is present and younger than
    :data:`LOCK_GRACE_SECONDS` (written by
    :func:`claim_gpus_for_training` before job.json exists, closing the
    claim->write race window).

    The active run's ``cuda_visible_devices`` decides enforcement via
    :func:`containers_to_stop`: the union over every active run stays
    stopped; every other configured container is (re)started; the pause
    sentinel stays set while any run is active. An unreadable job.json
    is treated conservatively -- keep every configured container
    stopped. Nothing active -> clear lock + sentinel, start everything.
    """
    sentinel_target = sentinel_path(sentinel)
    jobs_dir = train_jobs_dir if train_jobs_dir is not None else _resolve_train_jobs_dir()
    status_states: dict[str, str | None] = {}
    active_stems: set[str] = set()
    all_configured = tuple(name for name, _ in get_gpu_arbiter_config().container_gpus)
    stop_names: set[str] = set()

    def _accumulate(cvd: str | None) -> None:
        if cvd is None:
            stop_names.update(all_configured)
        else:
            stop_names.update(containers_to_stop(cvd))

    if jobs_dir.exists():
        for status_file in jobs_dir.glob('*.status.json'):
            try:
                payload = json.loads(status_file.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                continue
            stem = status_file.name[: -len('.status.json')]
            status_states[stem] = payload.get('state')

        for job_file in jobs_dir.glob('*.job.json'):
            stem = job_file.name[: -len('.job.json')]
            state = status_states.get(stem)
            # Active unless the status has reached a terminal state. No status
            # yet means just-claimed (still active); any non-terminal or
            # unrecognized state keeps the run live so an hours/days run can
            # never be misread as idle.
            if state is not None and state in TRAINER_TERMINAL_STATES:
                continue
            active_stems.add(stem)
            cvd: str | None = None
            try:
                cvd = json.loads(job_file.read_text(encoding='utf-8')).get('cuda_visible_devices')
            except (OSError, ValueError):
                cvd = None
            _accumulate(cvd)

    # The lock closes the window before job.json is visible, and ages out so
    # a crashed claim can't reserve the GPUs forever.
    lock = read_training_lock()
    if lock is not None:
        claimed_at = float(lock.get('claimed_at', 0.0) or 0.0)
        if (time.time() - claimed_at) < LOCK_GRACE_SECONDS:
            active_stems.add('__lock__')
            _accumulate(lock.get('cuda_visible_devices'))

    # A queued bake-off claims every configured container unless
    # OP_BAKEOFF_HOST_GPUS scopes it to the evaluator's host GPUs (then
    # only an intersecting container stays stopped).
    if bakeoff_active():
        active_stems.add('__bakeoff__')
        scope = get_gpu_arbiter_config().bakeoff_host_gpus
        stop_names.update(containers_to_stop(scope) if scope else all_configured)

    if active_stems:
        ordered_stop = tuple(name for name in all_configured if name in stop_names)
        to_start = tuple(name for name in all_configured if name not in stop_names)
        start_result: ArbiterAction | None = None
        if to_start:
            start_result = await start_gpu_services(containers=to_start)
        stop_result: ArbiterAction | None = None
        if ordered_stop:
            stop_result = await stop_gpu_services(containers=ordered_stop)
        # Reassert the sentinel last -- a run is active regardless of which
        # containers moved, so the paired worker must stay paused.
        pause_result = await pause_gpu_worker(sentinel=sentinel_target)
        logger.info(
            'arbiter_enforce_active',
            active_jobs=len(active_stems),
            stopped=list(ordered_stop),
            started=list(to_start),
        )
        if start_result is not None:
            return start_result
        if stop_result is not None:
            return stop_result
        return pause_result

    # Nothing active -- release everything: clear the (stale) lock +
    # sentinel and bring the configured containers back up. This is the
    # backstop that restarts services after a run finishes.
    clear_training_lock()
    await resume_gpu_worker(sentinel=sentinel_target)
    return await start_gpu_services()


# Re-exported for backward compat -- probe_trainer_reachable used to live
# in this module; split out to trainer_reachability.py to stay under the
# 700-LOC ratchet (see the comment above _container_status_sync). Imported
# at the bottom, after _container_status_sync/_docker_client are already
# bound in this module's namespace, so the circular import resolves.
from src.services.training.trainer_reachability import (  # noqa: E402
    TRAINER_CAPABILITIES_FILENAME,
    TRAINER_HEARTBEAT_STALE_SECONDS,
    probe_trainer_reachable,
)


__all__ = [
    'HEALTH_WAIT_SECONDS',
    'LOCK_GRACE_SECONDS',
    'TRAINER_CAPABILITIES_FILENAME',
    'TRAINER_HEARTBEAT_STALE_SECONDS',
    'TRAINER_TERMINAL_STATES',
    'ArbiterAction',
    'GpuArbiterStopFailedError',
    'bakeoff_active',
    'claim_gpus_for_training',
    'clear_training_lock',
    'containers_to_stop',
    'docker_client_available',
    'lock_path',
    'needs_multi_gpu_stop',
    'needs_service_stop',
    'parse_cuda_visible_devices',
    'pause_gpu_worker',
    'probe_trainer_reachable',
    'read_training_lock',
    'reconcile_on_startup',
    'release_gpus_after_training',
    'resume_gpu_worker',
    'sentinel_path',
    'set_training_lock',
    'start_gpu_services',
    'stop_gpu_services',
]
