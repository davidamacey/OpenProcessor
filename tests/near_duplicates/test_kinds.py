"""The near-duplicate job kinds: queue, exclusivity and auto-resume match the job plan."""

from __future__ import annotations

from src.config.job_kinds import KIND_BY_TASK, KINDS
from src.config.queue import TaskQueues
from src.config.task_routes import ROUTES


def test_the_four_kinds_follow_the_kind_table() -> None:
    expected = {
        'near_dup_score': ('near_dup', True, 86400),
        'near_dup_group': ('near_dup', False, 86400),
        'near_dup_rescore': ('near_dup', True, 3600),
        'near_dup_projection_sync': ('near_dup_projection', True, 86400),
    }
    for name, (group, resume, wait) in expected.items():
        kind = KINDS[name]
        assert (kind.exclusive_group, kind.exclusive_scope) == (group, 'level')
        assert (kind.auto_resume, kind.max_queue_wait_s) == (resume, wait)
        assert (kind.soft_limit_s, kind.hard_limit_s) == (10800, 11700)
        assert kind.project_scope == 'project'
        assert ROUTES[kind.task_name] == TaskQueues.JOBS
        assert KIND_BY_TASK[kind.task_name] is kind
