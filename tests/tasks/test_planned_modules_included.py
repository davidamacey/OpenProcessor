"""Every planned family module exists and is wired into ``include=`` and both aggregators.

So no later lane edits ``src/tasks/app.py`` or an aggregating ``__init__`` (review RR-4): a lane
only fills its own module.
"""

from __future__ import annotations

import importlib

import pytest

from src.config import job_kinds, task_routes
from src.tasks.app import TASK_MODULES


TASK_FAMILIES = (
    'infra',
    'reaper',
    'jobs_a',
    'jobs_b',
    'jobs_c',
    'trigger',
    'promote',
    'periodic',
    'b4_fanout',
    'b4_project',
    'beat_schedule',
    'clustering',
    'embedding_reindex',
    'near_dup',
    'near_dup_core',
    'source_backfill',
    'region_text',
)
# The lane that fills a module adds it here; the rest stay empty until their lane lands.
FILLED_RESEARCH_MODULES = frozenset({'clustering', 'near_dup', 'source_backfill'})
RESEARCH_TASK_MODULES = TASK_FAMILIES[-6:]
RESEARCH_CONFIG_MODULES = (
    'clustering',
    'b2_embedding',
    'near_dup',
    'near_dup_core',
    'source_backfill',
    'region_text',
)


@pytest.mark.parametrize('name', TASK_FAMILIES)
def test_every_task_module_exists_and_is_in_include(name: str) -> None:
    importlib.import_module(f'src.tasks.{name}')
    assert f'src.tasks.{name}' in TASK_MODULES


def test_include_lists_nothing_else() -> None:
    assert sorted(TASK_MODULES) == sorted(f'src.tasks.{name}' for name in TASK_FAMILIES)


@pytest.mark.parametrize('name', RESEARCH_CONFIG_MODULES)
def test_every_research_config_module_is_in_both_aggregators(name: str) -> None:
    kinds_module = importlib.import_module(f'src.config.job_kinds.{name}')
    routes_module = importlib.import_module(f'src.config.task_routes.{name}')
    assert kinds_module in job_kinds._MODULES
    assert routes_module in task_routes._MODULES
    if name not in FILLED_RESEARCH_MODULES:
        assert kinds_module.KINDS == ()
        assert routes_module.ROUTES == {}
