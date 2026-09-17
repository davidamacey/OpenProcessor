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
"""

from __future__ import annotations

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

    def is_gpu_allowed(self, gpu_id: int) -> bool:
        """``True`` unless ``allowed_gpu_ids`` is non-empty and excludes it."""
        if not self.allowed_gpu_ids:
            return True
        return gpu_id in self.allowed_gpu_ids


_default_gpu_arbiter_config: GpuArbiterConfig | None = None


def get_gpu_arbiter_config() -> GpuArbiterConfig:
    """Module-level default ``GpuArbiterConfig`` instance.

    Callers that need a deployment-specific instance (e.g. a future
    overlay pinning training to particular GPUs/containers) should
    construct and inject their own rather than relying on this default.
    """
    global _default_gpu_arbiter_config  # noqa: PLW0603 - lazily-built module singleton
    if _default_gpu_arbiter_config is None:
        _default_gpu_arbiter_config = GpuArbiterConfig()
    return _default_gpu_arbiter_config
