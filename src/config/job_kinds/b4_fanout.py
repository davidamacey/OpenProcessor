"""Job kinds of the `b4_fanout` family: batch routes that write by id and queue above ``OP_BATCH_SYNC_MAX``.

The two VLM batch kinds belong to the `trigger` family (they moved there with the VLM client).
"""

from __future__ import annotations

from src.config.job_kinds.kind import JobKind
from src.config.queue import TaskQueues


_HOUR_S = 3600
_QUARTER_S = 900


def _batch_kind(name: str) -> JobKind:
    """A 3 h kind that is its own exclusive group, project-wide (02b section 2.1)."""
    return JobKind(
        name=name,
        task_name=f'op.b4_fanout.{name}',
        queue=TaskQueues.JOBS,
        project_scope='project',
        max_queue_wait_s=_HOUR_S,
        exclusive_group=name,
        soft_limit_s=3 * _HOUR_S,
        hard_limit_s=3 * _HOUR_S + _QUARTER_S,
    )


KINDS: tuple[JobKind, ...] = (_batch_kind('review_resolve_batch'), _batch_kind('edit_undo_batch'))
