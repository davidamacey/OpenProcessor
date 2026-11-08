"""Job kinds of the `b2_embedding` research family (embeddings plan 5.5, job plan 02b section 2.1)."""

from __future__ import annotations

from src.config.job_kinds.kind import JobKind
from src.config.queue import TaskQueues


KINDS: tuple[JobKind, ...] = (
    # One build per space: the exclusive scope is the space key. slice_s keeps the job from
    # holding a `jobs` slot for hours (review RR-20): it yields every 10 minutes.
    JobKind(
        name='embedding_reindex',
        task_name='op.jobs.embedding_reindex',
        queue=TaskQueues.JOBS,
        project_scope='project',
        soft_limit_s=30 * 60,
        hard_limit_s=35 * 60,
        max_queue_wait_s=24 * 3600,
        exclusive_group='embedding_reindex',
        exclusive_scope='space_key',
        auto_resume=True,
        pausable=True,
        slice_s=600,
        gpu_affinity='any_card',
    ),
)
