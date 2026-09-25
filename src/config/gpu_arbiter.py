"""GPU-arbiter deployment configuration.

Three deployment-specific facts live here as data rather than as
hardcoded logic: which GPU ids a training job may target (a
module-level allowed-ids frozenset), which docker containers to
stop/start to free GPUs for a run (a module-level container-name
tuple), and the name of the trainer container to probe for
reachability (a module-level constant). All three are host/deployment
facts, not generic training-pipeline logic, so they live here rather
than in ``src/services/training/gpu_arbiter.py``.

**Default is permissive/empty, deliberately.** A generic OSS install has
no fixed GPU layout and no fixed set of sibling containers to coordinate
with, so:

- ``allowed_gpu_ids=frozenset()`` means *no restriction* — any GPU id the
  caller requests is accepted. A deployment that wants to pin training to
  specific GPUs (e.g. a two-GPU overlay restricting to ``{0, 2}``) sets
  this explicitly.
- ``containers=()`` means *nothing to stop/start* — GPU-arbiter container
  coordination becomes a no-op rather than an error when unconfigured
  (see ``tests/curation/test_gpu_arbiter_config.py``).
- ``trainer_container=None`` means *skip the reachability probe* — a
  generic install may not run a separate trainer container at all.

Every field is settable from the environment via
:meth:`GpuArbiterConfig.from_env` (``OP_GPU_ALLOWED_IDS``,
``OP_GPU_ARBITER_CONTAINERS``, ``OP_GPU_ARBITER_TRAINER_CONTAINER``,
``OP_BAKEOFF_JOBS_DIR``, ``OP_GPU_LABELS``, ``OP_TRAIN_DEFAULT_GPUS``),
which is what :func:`get_gpu_arbiter_config` builds the process-wide
default from — so a deployment pins its GPU policy in its env file, not
in code.

**GPU-scoped containers.** ``OP_GPU_ARBITER_CONTAINERS`` entries may
carry an optional GPU scope: ``name@2`` (single GPU) or ``name@0/2``
(``/`` separates ids within one entry's scope because ``,`` already
separates containers). A bare ``name`` is unscoped and keeps the
original semantics — stopped only when a claim spans more than one
GPU. A scoped container is stopped whenever the training claim
intersects its GPU set, regardless of claim size (see
:func:`src.services.training.gpu_arbiter.containers_to_stop`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _parse_gpu_scope(name: str, raw_ids: str) -> frozenset[int]:
    """Parse the ``ids`` half of a ``name@ids`` container entry (``/``-separated)."""
    tokens = [t.strip() for t in raw_ids.split('/')]
    if not any(tokens):
        msg = f'OP_GPU_ARBITER_CONTAINERS entry {name!r} has an empty GPU scope after "@"'
        raise ValueError(msg)
    ids: set[int] = set()
    for token in tokens:
        if not token:
            continue
        try:
            gpu_id = int(token)
        except ValueError as exc:
            msg = (
                f'OP_GPU_ARBITER_CONTAINERS entry {name!r} has a non-integer GPU scope id {token!r}'
            )
            raise ValueError(msg) from exc
        if gpu_id < 0:
            msg = f'OP_GPU_ARBITER_CONTAINERS entry {name!r} has a negative GPU scope id {token!r}'
            raise ValueError(msg)
        ids.add(gpu_id)
    return frozenset(ids)


def _parse_container_gpus(raw: str) -> tuple[tuple[str, frozenset[int] | None], ...]:
    """Parse ``OP_GPU_ARBITER_CONTAINERS`` into ``(name, gpu_scope | None)`` pairs.

    ``,`` separates container entries; each entry is either a bare
    ``name`` (unscoped, ``None``) or ``name@ids`` where ``ids`` is
    ``/``-separated (``name@2``, ``name@0/2``).
    """
    entries: list[tuple[str, frozenset[int] | None]] = []
    for raw_entry in raw.split(','):
        entry = raw_entry.strip()
        if not entry:
            continue
        if '@' in entry:
            name, _, scope = entry.partition('@')
            name = name.strip()
            if not name:
                msg = f'OP_GPU_ARBITER_CONTAINERS entry {entry!r} is missing a container name'
                raise ValueError(msg)
            entries.append((name, _parse_gpu_scope(name, scope)))
        else:
            entries.append((entry, None))
    return tuple(entries)


def _default_bakeoff_jobs_dir() -> str:
    """``<state_dir>/bakeoff_jobs``: the one default the router and arbiter share."""
    from src.config.curation import get_curation_config

    return str(get_curation_config().state_dir / 'bakeoff_jobs')


def _parse_gpu_labels(raw: str) -> dict[int, str]:
    """Parse ``OP_GPU_LABELS`` (``id=label,id=label,...``) into ``{id: label}``."""
    labels: dict[int, str] = {}
    for raw_entry in raw.split(','):
        entry = raw_entry.strip()
        if not entry:
            continue
        if '=' not in entry:
            msg = f'OP_GPU_LABELS entry {entry!r} is missing "="'
            raise ValueError(msg)
        raw_id, _, label = entry.partition('=')
        raw_id = raw_id.strip()
        label = label.strip()
        if not label:
            msg = f'OP_GPU_LABELS entry {entry!r} has an empty label'
            raise ValueError(msg)
        try:
            gpu_id = int(raw_id)
        except ValueError as exc:
            msg = f'OP_GPU_LABELS entry {entry!r} has a non-integer GPU id {raw_id!r}'
            raise ValueError(msg) from exc
        if gpu_id < 0:
            msg = f'OP_GPU_LABELS entry {entry!r} has a negative GPU id {raw_id!r}'
            raise ValueError(msg)
        labels[gpu_id] = label
    return labels


@dataclass(frozen=True)
class GpuArbiterConfig:
    """Deployment facts for :mod:`src.services.training.gpu_arbiter`.

    ``containers`` are stopped (in order) when a training/bake-off run
    claims all configured GPUs, and started (in the same order) when the
    run releases them. Empty by default — see module docstring.

    ``container_gpus`` is the parsed ``name -> gpu scope`` pairing (same
    order as ``containers``); ``None`` means unscoped (original
    "stop only on a multi-GPU claim" behavior). ``gpu_labels`` maps a
    GPU id to a human-readable card name for the served ``/train/gpus``
    options (unset ids just render as ``GPU {id}``). ``default_train_gpus``
    is the ``cuda_visible_devices`` value new training specs default to
    when the caller doesn't supply one.
    """

    allowed_gpu_ids: frozenset[int] = field(default_factory=frozenset)
    containers: tuple[str, ...] = ()
    container_gpus: tuple[tuple[str, frozenset[int] | None], ...] = ()
    trainer_container: str | None = None
    bakeoff_jobs_dir: str = field(default_factory=_default_bakeoff_jobs_dir)
    gpu_labels: dict[int, str] = field(default_factory=dict)
    default_train_gpus: str | None = None

    @classmethod
    def from_env(cls) -> GpuArbiterConfig:
        """Build a :class:`GpuArbiterConfig` from ``OP_*`` env vars.

        - ``OP_GPU_ALLOWED_IDS`` — comma-separated GPU ids a training /
          bake-off job may target (e.g. ``0,2``). Unset or empty keeps
          the permissive default (any id).
        - ``OP_GPU_ARBITER_CONTAINERS`` — comma-separated container names,
          each optionally scoped to specific GPU ids with ``name@ids``
          (``ids`` is ``/``-separated, e.g. ``vllm-server@2`` or
          ``vllm-server@0/2``). Scoped containers are stopped (in order)
          whenever a training claim intersects their GPU set; unscoped
          entries keep the original behavior (stopped only when a claim
          spans more than one GPU). Restarted (in the same order) when the
          run releases the GPUs. Unset/empty = nothing to coordinate.
        - ``OP_GPU_ARBITER_TRAINER_CONTAINER`` — trainer container name to
          probe for reachability. Unset/empty = skip the probe.
        - ``OP_BAKEOFF_JOBS_DIR`` — the bake-off job-queue directory the
          reconcile loop watches and the bake-off router writes job files
          into (the router reads this field, so the two cannot drift
          apart). Unset = ``<CurationConfig.state_dir>/bakeoff_jobs``,
          never ``None``: a queued bake-off must keep GPU-resident
          containers stopped on the default config too.
        - ``OP_GPU_LABELS`` — comma-separated ``id=label`` pairs (e.g.
          ``0=RTX A6000,2=RTX A6000``) used to build human-readable
          ``/train/gpus`` option labels. Unset/empty = no labels (options
          fall back to ``GPU {id}``).
        - ``OP_TRAIN_DEFAULT_GPUS`` — the ``cuda_visible_devices`` value a
          new training spec defaults to when the caller omits it. Unset =
          derive from ``allowed_gpu_ids`` (see
          ``src.services.training.jobs.default_train_gpu_value``).

        A malformed ``OP_GPU_ALLOWED_IDS`` (non-integer or negative token),
        ``OP_GPU_ARBITER_CONTAINERS`` scope (non-integer or negative id, or
        an empty scope after ``@``), or ``OP_GPU_LABELS`` entry (missing
        ``=``, non-integer or negative id) raises ``ValueError`` rather than
        silently degrading — a typo in a GPU fence must not remove the
        fence.
        """
        raw_ids = os.environ.get('OP_GPU_ALLOWED_IDS', '')
        ids: set[int] = set()
        for token in (t.strip() for t in raw_ids.split(',')):
            if not token:
                continue
            try:
                gpu_id = int(token)
            except ValueError as exc:
                msg = (
                    f'OP_GPU_ALLOWED_IDS must be a comma-separated list of GPU ids, got {raw_ids!r}'
                )
                raise ValueError(msg) from exc
            if gpu_id < 0:
                msg = f'OP_GPU_ALLOWED_IDS must not contain negative ids, got {raw_ids!r}'
                raise ValueError(msg)
            ids.add(gpu_id)
        container_gpus = _parse_container_gpus(os.environ.get('OP_GPU_ARBITER_CONTAINERS', ''))
        containers = tuple(name for name, _ in container_gpus)
        trainer = os.environ.get('OP_GPU_ARBITER_TRAINER_CONTAINER', '').strip() or None
        jobs_dir = os.environ.get('OP_BAKEOFF_JOBS_DIR', '').strip() or _default_bakeoff_jobs_dir()
        gpu_labels = _parse_gpu_labels(os.environ.get('OP_GPU_LABELS', ''))
        default_train_gpus = os.environ.get('OP_TRAIN_DEFAULT_GPUS', '').strip() or None
        return cls(
            allowed_gpu_ids=frozenset(ids),
            containers=containers,
            container_gpus=container_gpus,
            trainer_container=trainer,
            bakeoff_jobs_dir=jobs_dir,
            gpu_labels=gpu_labels,
            default_train_gpus=default_train_gpus,
        )

    def is_gpu_allowed(self, gpu_id: int) -> bool:
        """``True`` unless ``allowed_gpu_ids`` is non-empty and excludes it."""
        if not self.allowed_gpu_ids:
            return True
        return gpu_id in self.allowed_gpu_ids


_default_gpu_arbiter_config: GpuArbiterConfig | None = None


def get_gpu_arbiter_config() -> GpuArbiterConfig:
    """Module-level default ``GpuArbiterConfig`` instance.

    Built via :meth:`GpuArbiterConfig.from_env` on first use, so the
    ``OP_GPU_*`` / ``OP_BAKEOFF_JOBS_DIR`` env vars take effect for every
    consumer (the lifespan reconcile loop, training-job GPU validation,
    the bake-off router). Like the other ``OP_*`` config it is resolved
    once per process — set the env before startup.
    """
    global _default_gpu_arbiter_config  # noqa: PLW0603 - lazily-built module singleton
    if _default_gpu_arbiter_config is None:
        _default_gpu_arbiter_config = GpuArbiterConfig.from_env()
    return _default_gpu_arbiter_config
