"""Job kinds of lane 4C: ingest and project-delete families (02b section 2.1)."""

from __future__ import annotations

from src.config.job_kinds.kind import JobKind
from src.config.queue import TaskQueues


HOUR = 3600

KINDS: tuple[JobKind, ...] = (
    JobKind(
        name='project_delete',
        task_name='op.families_c.project_delete',
        queue=TaskQueues.JOBS,
        project_scope='deployment',
        soft_limit_s=3 * HOUR,
        hard_limit_s=3 * HOUR + 900,
        max_queue_wait_s=24 * HOUR,
        exclusive_group='project_delete',
        exclusive_scope='slug',
        auto_resume=True,
    ),
    JobKind(
        name='open_vocab_drain',
        task_name='op.families_c.open_vocab_drain',
        queue=TaskQueues.OPEN_VOCAB,
        project_scope='project',
        soft_limit_s=3 * HOUR,
        hard_limit_s=3 * HOUR + 900,
        max_queue_wait_s=HOUR,
        exclusive_group='open_vocab_drain',
    ),
    JobKind(
        name='directory_ingest',
        task_name='op.families_c.directory_ingest',
        queue=TaskQueues.JOBS,
        project_scope='deployment',
        soft_limit_s=24 * HOUR,
        hard_limit_s=24 * HOUR + 900,
        max_queue_wait_s=24 * HOUR,
        gpu_affinity='any_card',
    ),
    JobKind(
        name='bulk_ingest_run',
        task_name='op.families_c.bulk_ingest_run',
        queue=TaskQueues.JOBS,
        project_scope='project',
        soft_limit_s=24 * HOUR,
        hard_limit_s=24 * HOUR + 900,
        max_queue_wait_s=24 * HOUR,
        exclusive_group='bulk_ingest',
        auto_resume=True,
        gpu_affinity='any_card',
    ),
    JobKind(
        name='source_poll',
        task_name='op.families_c.source_poll',
        queue=TaskQueues.JOBS,
        project_scope='project',
        soft_limit_s=HOUR,
        hard_limit_s=HOUR + 300,
        max_queue_wait_s=HOUR,
        exclusive_group='source_poll',
        exclusive_scope='source_id',
    ),
)
