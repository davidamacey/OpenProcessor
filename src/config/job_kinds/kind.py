"""The ``JobKind`` record: one row of the kind table in the job plan (02b section 2.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.config.queue import TaskQueues


ProjectScope = Literal['project', 'deployment', 'target', 'either']
GpuAffinity = Literal['none', 'any_card']
GPU_AFFINITIES: tuple[str, ...] = ('none', 'any_card')
MAX_ATTEMPTS = 3  # D-20: every kind
# The longest wait for a slice boundary is one VLM request; a sliced kind's soft limit must
# clear the slice plus this, or the limit would fire at a boundary.
VLM_REQUEST_TIMEOUT_S = 300


@dataclass(frozen=True)
class JobKind:
    """Pure data. ``register_job`` and ``start_job`` read queue, limits and auto-resume here.

    ``exclusive_group`` / ``exclusive_scope``: at most one active job per (project, group,
    scope value); the scope names the ``params`` key whose value narrows the group, ``''``
    meaning the whole project. ``None`` means no exclusivity.
    """

    name: str
    task_name: str
    queue: str
    project_scope: ProjectScope
    soft_limit_s: int
    hard_limit_s: int
    max_queue_wait_s: int
    exclusive_group: str | None = None
    exclusive_scope: str = ''
    auto_resume: bool = False
    max_attempts: int = MAX_ATTEMPTS
    pausable: bool = False
    slice_s: int | None = None
    gpu_affinity: GpuAffinity = 'none'

    def __post_init__(self) -> None:
        if self.queue not in TaskQueues.ALL:
            raise ValueError(f'job kind {self.name}: unknown queue {self.queue!r}')
        if not 0 < self.soft_limit_s < self.hard_limit_s:
            raise ValueError(f'job kind {self.name}: need 0 < soft limit < hard limit')
        if self.gpu_affinity not in GPU_AFFINITIES:
            raise ValueError(f'job kind {self.name}: unknown gpu_affinity {self.gpu_affinity!r}')
        if self.slice_s is not None:
            if not self.pausable:
                raise ValueError(f'job kind {self.name}: slice_s needs pausable=True')
            if self.soft_limit_s <= self.slice_s + VLM_REQUEST_TIMEOUT_S:
                raise ValueError(
                    f'job kind {self.name}: soft limit {self.soft_limit_s} s must exceed '
                    f'slice_s {self.slice_s} s plus one VLM request ({VLM_REQUEST_TIMEOUT_S} s)'
                )
