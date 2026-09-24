"""GPU-arbiter deployment configuration.

The reference implementation hardcodes three deployment-specific facts
directly in code: which GPU ids a training job may target (a module-level
allowed-ids frozenset), which docker containers to stop/start to free
GPUs for a run (a module-level container-name tuple), and the name of
the trainer container to probe for reachability (a module-level
constant). All three are host/deployment facts, not generic
training-pipeline logic, so they live here as data rather than in
``src/services/training/gpu_arbiter.py``.

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
``OP_BAKEOFF_JOBS_DIR``), which is what :func:`get_gpu_arbiter_config`
builds the process-wide default from — so a deployment pins its GPU
policy in its env file, not in code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GpuArbiterConfig:
    """Deployment facts for :mod:`src.services.training.gpu_arbiter`.

    ``containers`` are stopped (in order) when a training/bake-off run
    claims all configured GPUs, and started (in the same order) when the
    run releases them. Empty by default — see module docstring.
    """

    allowed_gpu_ids: frozenset[int] = field(default_factory=frozenset)
    containers: tuple[str, ...] = ()
    trainer_container: str | None = None
    bakeoff_jobs_dir: str | None = None

    @classmethod
    def from_env(cls) -> GpuArbiterConfig:
        """Build a :class:`GpuArbiterConfig` from ``OP_*`` env vars.

        - ``OP_GPU_ALLOWED_IDS`` — comma-separated GPU ids a training /
          bake-off job may target (e.g. ``0,2``). Unset or empty keeps
          the permissive default (any id).
        - ``OP_GPU_ARBITER_CONTAINERS`` — comma-separated container names
          stopped (in order) when a run claims the GPUs and restarted
          when it releases them. Unset/empty = nothing to coordinate.
        - ``OP_GPU_ARBITER_TRAINER_CONTAINER`` — trainer container name to
          probe for reachability. Unset/empty = skip the probe.
        - ``OP_BAKEOFF_JOBS_DIR`` — the bake-off job-queue directory the
          reconcile loop watches; the same var the bake-off router writes
          job files into, so the two cannot drift apart.

        A malformed ``OP_GPU_ALLOWED_IDS`` (non-integer or negative token)
        raises ``ValueError`` rather than silently degrading to
        "unrestricted" — a typo in a GPU fence must not remove the fence.
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
        containers = tuple(
            name.strip()
            for name in os.environ.get('OP_GPU_ARBITER_CONTAINERS', '').split(',')
            if name.strip()
        )
        trainer = os.environ.get('OP_GPU_ARBITER_TRAINER_CONTAINER', '').strip() or None
        jobs_dir = os.environ.get('OP_BAKEOFF_JOBS_DIR', '').strip() or None
        return cls(
            allowed_gpu_ids=frozenset(ids),
            containers=containers,
            trainer_container=trainer,
            bakeoff_jobs_dir=jobs_dir,
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
