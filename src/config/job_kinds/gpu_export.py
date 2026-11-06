"""Job kinds of the `gpu_export` family: the model upload and export build (02b section 2.1)."""

from __future__ import annotations

from src.config.job_kinds.kind import JobKind
from src.config.queue import TaskQueues


KINDS: tuple[JobKind, ...] = (
    JobKind(
        name='model_export',
        task_name='op.export.model_export',
        queue=TaskQueues.GPU_EXPORT,
        project_scope='either',
        soft_limit_s=4 * 3600,
        hard_limit_s=4 * 3600 + 900,
        max_queue_wait_s=24 * 3600,
        exclusive_group='model_export',
        exclusive_scope='triton_name',
    ),
)
