"""Label operations a person waits on run on their own queue, so bulk jobs on ``jobs`` (project
deletes, ingest, reprocess) cannot hold every slot in front of them.

Docs: https://docs.celeryq.dev/en/stable/userguide/routing.html (``task_routes``, one queue per
worker pool)."""

from __future__ import annotations

from src.config.job_kinds import KINDS
from src.config.queue import QUEUE_PROFILE, TaskQueues
from src.config.task_routes import ROUTES, celery_task_routes


def test_label_ops_are_routed_to_the_interactive_queue() -> None:
    for task in ('op.jobs_a.label_op', 'op.jobs_a.label_undo'):
        assert ROUTES[task] == TaskQueues.INTERACTIVE
        assert celery_task_routes()[task] == {'queue': 'interactive'}


def test_the_job_kind_and_the_route_name_the_same_queue() -> None:
    assert KINDS['label_op'].queue == KINDS['label_undo'].queue == TaskQueues.INTERACTIVE


def test_the_interactive_queue_has_a_core_consumer_and_only_label_ops() -> None:
    assert QUEUE_PROFILE[TaskQueues.INTERACTIVE] == 'core'
    assert sorted(t for t, q in ROUTES.items() if q == TaskQueues.INTERACTIVE) == [
        'op.jobs_a.label_op',
        'op.jobs_a.label_undo',
    ]
