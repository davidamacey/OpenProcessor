"""The kind table rows of the trigger family match the jobs plan (02b section 2.1)."""

from __future__ import annotations

import pytest

from src.config.job_kinds import KINDS
from src.config.queue import TaskQueues
from src.config.task_routes import ROUTES


def test_autolabel_is_pausable_unsliced_and_exclusive_per_project() -> None:
    kind = KINDS['autolabel']
    assert (kind.queue, kind.pausable, kind.slice_s) == (TaskQueues.AUTOLABEL, True, None)
    assert (kind.soft_limit_s, kind.hard_limit_s) == (10800, 11700)
    assert (kind.exclusive_group, kind.exclusive_scope, kind.auto_resume) == (
        'autolabel',
        '',
        False,
    )
    assert kind.max_queue_wait_s == 3600


@pytest.mark.parametrize('name', ['vlm_label_batch', 'vlm_verify_region_batch'])
def test_the_vlm_batch_kinds_are_plain_jobs(name: str) -> None:
    kind = KINDS[name]
    assert (kind.queue, kind.pausable, kind.slice_s, kind.auto_resume) == (
        TaskQueues.JOBS,
        False,
        None,
        False,
    )
    assert (kind.soft_limit_s, kind.hard_limit_s, kind.max_queue_wait_s) == (10800, 11700, 3600)


def test_every_trigger_kind_has_exactly_its_route() -> None:
    for name in ('autolabel', 'vlm_label_batch', 'vlm_verify_region_batch'):
        kind = KINDS[name]
        assert ROUTES[kind.task_name] == kind.queue
