"""Job kinds of the `gpu_eval` family: the model bake-off, run by the evaluator image."""

from __future__ import annotations

from src.config.job_kinds.kind import JobKind
from src.config.queue import TaskQueues


KINDS: tuple[JobKind, ...] = (
    JobKind(
        name='bakeoff',
        task_name='op.gpu_eval.bakeoff',
        queue=TaskQueues.GPU_EVAL,
        project_scope='project',
        soft_limit_s=24 * 3600,
        hard_limit_s=24 * 3600 + 900,
        max_queue_wait_s=24 * 3600,
    ),
)
