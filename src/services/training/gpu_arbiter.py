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

import dataclasses
import json
import time
from pathlib import Path

from src.config import get_gpu_arbiter_config
from src.core.logging import get_logger
from src.services.training.arbiter_dirs import all_bakeoff_jobs_dirs, all_train_jobs_dirs
from src.services.training.gpu_arbiter_docker import start_gpu_services, stop_gpu_services
from src.services.training.gpu_arbiter_policy import (
    ArbiterAction,
    GpuArbiterStopFailedError,
    clear_training_lock,
    containers_to_stop,
    pause_gpu_worker,
    read_training_lock,
    resume_gpu_worker,
    sentinel_path,
    set_training_lock,
)


logger = get_logger(__name__)


HEALTH_WAIT_SECONDS = 90

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

    ``jobs_dir`` scans only that dir. By default every dir a bake-off can
    be queued in is scanned (:func:`all_bakeoff_jobs_dirs`): the arbiter is
    global, but the bake-off router writes into the bound project's own
    ``bakeoff_jobs_dir`` (projects_plan.md §5.3). A missing dir means
    nothing queued.
    """
    targets = [Path(jobs_dir)] if jobs_dir is not None else all_bakeoff_jobs_dirs()
    for target in targets:
        try:
            if any(target.glob('*.job.json')):
                return True
        except OSError:
            continue
    return False


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


async def reconcile_on_startup(
    *,
    train_jobs_dir: Path | None = None,
    sentinel: Path | None = None,
) -> ArbiterAction:
    """Enforce the desired GPU-service state -- both directions.

    ``train_jobs_dir`` defaults to :func:`all_train_jobs_dirs` (default
    dir + every active/archived project's own dir, §5.3), so a run
    started in any project keeps the GPUs claimed. An explicit path
    (test fixtures) scans only that one dir, attributed to no project.

    Runs once at API startup *and* on a periodic loop (every uvicorn
    worker); idempotent re-enforcement, not a race.

    A run owns the GPUs when a ``*.job.json`` exists whose
    ``*.status.json`` hasn't reached a :data:`TRAINER_TERMINAL_STATES`
    state, or the training lock is present and younger than
    :data:`LOCK_GRACE_SECONDS`. Enforcement via ``cuda_visible_devices``
    + :func:`containers_to_stop`; nothing active -> clear lock/sentinel
    and start everything.
    """
    sentinel_target = sentinel_path(sentinel)
    if train_jobs_dir is not None:
        jobs_dirs: dict[str, Path] = {'': train_jobs_dir}
    else:
        jobs_dirs = all_train_jobs_dirs()
    status_states: dict[str, str | None] = {}
    active_stems: set[str] = set()
    all_configured = tuple(name for name, _ in get_gpu_arbiter_config().container_gpus)
    stop_names: set[str] = set()
    holder_project: str | None = None

    def _accumulate(cvd: str | None) -> None:
        if cvd is None:
            stop_names.update(all_configured)
        else:
            stop_names.update(containers_to_stop(cvd))

    for project_slug, jobs_dir in jobs_dirs.items():
        if not jobs_dir.exists():
            continue
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
            if holder_project is None and project_slug:
                holder_project = project_slug

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
        result = start_result if start_result is not None else stop_result
        if result is None:
            result = pause_result
        return dataclasses.replace(result, holder_project=holder_project)

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
    'bakeoff_active',
    'claim_gpus_for_training',
    'reconcile_on_startup',
    'release_gpus_after_training',
]
