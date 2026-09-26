"""Regression test: detect_cluster_backend() must never be called
synchronously from async code.

Real bug, observed live: cupy.cuda.runtime.getDeviceCount() (inside
detect_cluster_backend) blocked ~30+ seconds hitting a driver-mismatch
error in a container with no GPU device nodes at all (the auto-label
worker is deliberately CPU-only). Called as a plain synchronous function
from async code, that starved the worker's sibling heartbeat coroutine
long enough for the API's 30s staleness watchdog to kill a genuinely-
running clustering job ("auto_label worker heartbeat stale (38.0s ago)"),
despite the function's own docstring claiming "a few microseconds."

Both real call sites now wrap the call as `asyncio.to_thread(detect_
cluster_backend)` — the function is passed *by reference* there (no
parens, no direct invocation), so a correctly-fixed module has zero
`ast.Call` nodes with func name `detect_cluster_backend`. Only the
buggy `detect_cluster_backend()` direct-call form produces one — which
is exactly what this asserts against, walking the real module source
rather than re-testing asyncio.to_thread's own (already well-established)
behavior.
"""

from __future__ import annotations

import ast
import inspect

from src.services.curation.clustering import embedding_reduce, orchestrator


def _synchronous_call_lines(module) -> list[int]:
    tree = ast.parse(inspect.getsource(module))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == 'detect_cluster_backend'
    ]


def test_embedding_reduce_never_calls_detect_cluster_backend_synchronously():
    offending = _synchronous_call_lines(embedding_reduce)
    assert offending == [], (
        f'detect_cluster_backend() called synchronously (blocks the event '
        f'loop — see module docstring) at embedding_reduce.py line(s) '
        f'{offending}; must be asyncio.to_thread(detect_cluster_backend)'
    )


def test_clustering_orchestrator_never_calls_detect_cluster_backend_synchronously():
    offending = _synchronous_call_lines(orchestrator)
    assert offending == [], (
        f'detect_cluster_backend() called synchronously (blocks the event '
        f'loop — see module docstring) at orchestrator.py line(s) '
        f'{offending}; must be asyncio.to_thread(detect_cluster_backend)'
    )
