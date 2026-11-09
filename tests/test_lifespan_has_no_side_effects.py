"""The API lifespan starts no periodic work: beat ticks do that now (``src/tasks/periodic.py``).

An AST read of ``src/app/lifespan.py``: none of the removed periodic steps may come back, whatever
their form (a call, an import, an attribute). The ingest dispatcher loop, the open-vocabulary
sweeper, the promoted-model reload, the orphaned-job reconcile, the cache prune and the GPU arbiter
(``op.periodic.gpu_arbiter``, in gpu-arbiter-worker) each run as a
tick, so the number of API workers never multiplies them.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


LIFESPAN = Path(__file__).resolve().parents[1] / 'src' / 'app' / 'lifespan.py'
REMOVED = (
    'start_background_jobs',
    'stop_background_jobs',
    'background_jobs',
    'curation_background_jobs',
    'start_ingest_dispatcher',
    'start_open_vocab_sweeper',
    'reconcile_orphaned',
    'startup_reconcile',
    'reload_promoted_models',
    'prune_crop_cache',
    'start_gpu_arbiter',
    'arbiter_reconcile_loop',
    'stop_arbiter',
    'reconcile_on_startup',
    'arbiter_task',
)


def mentioned_names() -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(LIFESPAN.read_text(encoding='utf-8'))):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.alias):
            names.add(node.name.rsplit('.', 1)[-1])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.update(node.module.split('.'))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
    return names


@pytest.mark.parametrize('removed', REMOVED)
def test_a_removed_periodic_step_is_not_in_the_lifespan(removed: str) -> None:
    assert not any(removed in name for name in mentioned_names())


def test_the_probe_sees_what_is_there() -> None:
    assert 'lifespan' in mentioned_names()
