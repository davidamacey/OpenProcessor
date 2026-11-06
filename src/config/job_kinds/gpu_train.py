"""Job kinds of the `gpu_train` family: one training run on the trainer's card (02b section 2.1)."""

from __future__ import annotations

from src.config.job_kinds.kind import JobKind
from src.config.queue import TaskQueues


KINDS: tuple[JobKind, ...] = (
    JobKind(
        name='train',
        task_name='op.gpu_train.train',
        queue=TaskQueues.GPU_TRAIN,
        project_scope='project',
        soft_limit_s=72 * 3600,
        hard_limit_s=72 * 3600 + 900,
        max_queue_wait_s=7 * 24 * 3600,
    ),
)
