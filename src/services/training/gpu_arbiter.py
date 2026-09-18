"""GPU arbiter — coordinate GPU-resident services around training runs.

Ported from a private reference vehicle/license-plate curation stack's
training pipeline (Phase 3a). Two regimes:

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
every tick: while the lock / an active job owns the GPUs it re-stops any
service that comes back up (blocking the hand-off), and once nothing is
active it clears the lock and restarts the services — which doubles as
crash recovery.

**Deployment facts vs. mechanism.** Which GPU ids a job may target, which
containers to stop/start, and which container is "the trainer" for
reachability probing are deployment facts — see
:class:`src.config.gpu_arbiter.GpuArbiterConfig`. Every public function
here takes them as parameters with a config-derived default, so a generic
install with nothing configured degrades to a no-op rather than crashing
(see ``tests/curation/test_gpu_arbiter_config.py``).

Implementation note: container control uses the docker SDK over the
mounted host socket (``/var/run/docker.sock``), stopping/starting
containers by name so restart preserves each container's original config
(GPU pins included). When the SDK/socket is unavailable it falls back to
the sentinel-pause path only and logs a warning; single-GPU runs still
work.
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


# =============================================================================
# Constants
# =============================================================================


def _state_dir() -> Path:
    return get_curation_config().state_dir


def _default_sentinel_path() -> Path:
    return _state_dir() / 'training_worker' / 'pause.sentinel'


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

    ``jobs_dir`` defaults to ``GpuArbiterConfig.bakeoff_jobs_dir``; when
    that is unset (the generic-install default), there is no bake-off
    harness wired up to watch, so this always reports ``False``.
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
    """Does this claim require stopping GPU-resident services entirely?

    True whenever the claim spans more than one GPU id. A single-GPU claim
    leaves the other configured GPU(s) untouched, so only the paired
    worker's sentinel is set (see :func:`pause_gpu_worker`).
    """
    devices = parse_cuda_visible_devices(cuda_visible_devices)
    return len(devices) > 1


# Back-compat alias for the reference implementation's name.
needs_gemma_stop = needs_multi_gpu_stop


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
    """Return a docker SDK client over the mounted socket, or ``None``.

    Imported lazily so the module stays importable in environments
    without the ``docker`` package or the socket (unit tests, hosts with
    no GPU-resident sibling containers). Any failure (no package, no
    socket, no permission) returns ``None`` so callers fall back to the
    sentinel-only path.
    """
    try:
        import docker  # type: ignore[import-untyped]

        client = docker.from_env()  # type: ignore[attr-defined]
        client.ping()
        return client
    except Exception as exc:
        logger.warning('arbiter_docker_unavailable', error=str(exc))
        return None


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

    ``containers`` defaults to ``GpuArbiterConfig.containers``. If that is
    empty (the generic-install default), this is a pure no-op -- it does
    not touch the docker SDK at all, so an unconfigured install never
    logs a spurious "docker unavailable" warning.

    Otherwise uses the docker SDK over the mounted socket; restart later
    preserves each container's original config (GPU pins included).
    Falls back to the sentinel-only pause if the socket/SDK is
    unavailable.
    """
    names = containers if containers is not None else get_gpu_arbiter_config().containers
    if not names:
        return ArbiterAction(action='noop', detail='no GPU-resident containers configured')
    client = _docker_client()
    if client is None:
        await pause_gpu_worker()
        return ArbiterAction(
            action='sentinel_set',
            detail='docker SDK unavailable; fell back to sentinel-only',
        )
    try:
        stopped = await asyncio.to_thread(_stop_containers_sync, client, names)
    except Exception as exc:
        logger.warning('arbiter_gpu_stop_failed', error=str(exc))
        await pause_gpu_worker()
        return ArbiterAction(action='sentinel_set', detail=f'stop failed: {exc}; sentinel set')
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


def _container_status_sync(client: Any, name: str) -> str | None:
    """Return the named container's status, or ``None`` if it doesn't exist."""
    import docker.errors

    try:
        container = client.containers.get(name)
    except docker.errors.NotFound:
        return None
    container.reload()
    return container.status


async def probe_trainer_reachable(
    *,
    container_name: str | None = None,
) -> tuple[bool, str]:
    """Check whether the configured trainer container is up and running.

    Without this, submitting a training job can write ``job.json`` and
    have the run sit in ``queued`` forever with no error if the trainer
    container was never started. Preflight calls this so submitting into
    a void is a blocking failure instead of a silent forever-queue.

    ``container_name`` defaults to ``GpuArbiterConfig.trainer_container``.
    When that is unset (the generic-install default -- no separate
    trainer container to probe), this returns ``(True, ...)`` rather than
    treating "not configured" as a failure: a generic install may run
    training in-process with no sibling container at all.

    Returns ``(reachable, detail)``. Any failure to determine the real
    state -- missing docker SDK, no socket, permission error -- reports
    ``reachable=False`` rather than silently passing: we can't tell
    "trainer is fine" from "we can't see it," and the whole point of this
    check is to not queue into an unknown.
    """
    name = (
        container_name if container_name is not None else get_gpu_arbiter_config().trainer_container
    )
    if not name:
        return True, 'no trainer container configured -- reachability probe not applicable'
    client = _docker_client()
    if client is None:
        return (
            False,
            f'docker SDK/socket unavailable in the API container -- cannot verify {name!r} is running',
        )
    try:
        state = await asyncio.to_thread(_container_status_sync, client, name)
    except Exception as exc:
        return False, f'{name!r} probe failed: {exc}'
    if state is None:
        return False, f'{name!r} container does not exist'
    if state != 'running':
        return False, f'{name!r} container status={state!r} (expected running)'
    return True, f'{name!r} is running'


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
    """
    set_training_lock(cuda_visible_devices)
    if needs_multi_gpu_stop(cuda_visible_devices):
        return await stop_gpu_services()
    return await pause_gpu_worker()


async def release_gpus_after_training(
    cuda_visible_devices: str | None,
) -> ArbiterAction:
    """Inverse of :func:`claim_gpus_for_training`.

    For a multi-GPU run this restarts the configured containers; for a
    single-GPU run it just clears the pause sentinel. Idempotent -- safe
    to call when nothing was stopped (e.g. crash before claim). In
    practice the API's reconcile loop (:func:`reconcile_on_startup`) is
    the backstop that restarts services once no run is active, since the
    trainer container may have no docker socket.
    """
    clear_training_lock()
    if needs_multi_gpu_stop(cuda_visible_devices):
        return await start_gpu_services()
    return await resume_gpu_worker()


# ---- recovery on API startup -------------------------------------------


async def reconcile_on_startup(
    *,
    train_jobs_dir: Path = Path('/jobs'),
    sentinel: Path | None = None,
) -> ArbiterAction:
    """Enforce the desired GPU-service state -- both directions.

    This runs once at API startup *and* on a periodic loop, in every
    uvicorn worker. It is the single authority that decides whether the
    configured GPU-resident containers should be **down** (a training
    run owns the GPUs) or **up** (no run active), and it actively drives
    the containers toward that state on every tick. Because
    :func:`stop_gpu_services` / :func:`start_gpu_services` only act on
    containers not already in the target state, running this in all
    workers is harmless idempotent re-enforcement, not a race.

    A run is considered to own the GPUs when **either**:

    * a ``*.job.json`` exists whose ``*.status.json`` has *not* reached a
      :data:`TRAINER_TERMINAL_STATES` state (no status yet = just-claimed;
      any non-terminal/unknown state = still live, so a multi-day
      ``running`` run is never misread as idle); **or**
    * the training lock is present and younger than
      :data:`LOCK_GRACE_SECONDS`. The lock is written by
      :func:`claim_gpus_for_training` *before* job.json exists, closing
      the claim->write window where the file-based check alone would see
      "idle".

    The active run's ``cuda_visible_devices`` (read from job.json, or the
    lock) decides the enforcement: a **multi-GPU** run keeps the
    configured containers stopped; a **single-GPU** run only keeps the
    worker paused. When nothing is active we clear the lock + sentinel
    and start the containers back up.
    """
    sentinel_target = sentinel_path(sentinel)
    status_states: dict[str, str | None] = {}
    active_stems: set[str] = set()
    active_multi = False
    active_single = False

    if train_jobs_dir.exists():
        for status_file in train_jobs_dir.glob('*.status.json'):
            try:
                payload = json.loads(status_file.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                continue
            stem = status_file.name[: -len('.status.json')]
            status_states[stem] = payload.get('state')

        for job_file in train_jobs_dir.glob('*.job.json'):
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
            # Unknown device set -> assume multi-GPU (conservative: keep the
            # configured containers free rather than risk contending with a
            # live run).
            if cvd is None or needs_multi_gpu_stop(cvd):
                active_multi = True
            else:
                active_single = True

    # The lock closes the window before job.json is visible, and ages out so
    # a crashed claim can't reserve the GPUs forever.
    lock = read_training_lock()
    if lock is not None:
        claimed_at = float(lock.get('claimed_at', 0.0) or 0.0)
        if (time.time() - claimed_at) < LOCK_GRACE_SECONDS:
            active_stems.add('__lock__')
            if needs_multi_gpu_stop(lock.get('cuda_visible_devices')):
                active_multi = True
            else:
                active_single = True

    # A queued/running bake-off claims the same containers as a multi-GPU
    # train (only meaningful when a bake-off jobs dir is configured).
    if bakeoff_active():
        active_stems.add('__bakeoff__')
        active_multi = True

    if active_multi:
        # BLOCKING: keep the configured containers down for the whole run.
        # Stopping an already-stopped container is a no-op, so this is
        # cheap steady-state.
        logger.info('arbiter_enforce_stopped', active_jobs=len(active_stems), mode='multi')
        return await stop_gpu_services()

    if active_single:
        # Single-GPU run: only the worker stays paused; containers keep
        # running on their own GPU. Re-assert the sentinel in case a
        # worker cleared it.
        logger.info('arbiter_enforce_paused', active_jobs=len(active_stems), mode='single')
        return await pause_gpu_worker(sentinel=sentinel_target)

    # Nothing active -- release everything: clear the (stale) lock +
    # sentinel and bring the configured containers back up. This is the
    # backstop that restarts services after a run finishes.
    clear_training_lock()
    await resume_gpu_worker(sentinel=sentinel_target)
    return await start_gpu_services()


__all__ = [
    'HEALTH_WAIT_SECONDS',
    'LOCK_GRACE_SECONDS',
    'TRAINER_TERMINAL_STATES',
    'ArbiterAction',
    'bakeoff_active',
    'claim_gpus_for_training',
    'clear_training_lock',
    'lock_path',
    'needs_gemma_stop',
    'needs_multi_gpu_stop',
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
