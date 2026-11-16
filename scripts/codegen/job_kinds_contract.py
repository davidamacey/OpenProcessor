"""Render ``contracts/json/job_kinds.json``: what the Rust job runner reads from the Python job tables.

The runner (``native/op_jobs``) keeps no copy of the kind table, the state machine or the lease
timing. Everything it needs comes from here, from the same modules the Python wrapper uses:

* ``kinds``: every row of ``src/config/job_kinds/`` with all the fields of ``JobKind`` (name, task
  name, queue, project scope, limits, queue wait, exclusive group and scope, auto-resume, attempts,
  pausable, slice, GPU affinity) plus ``runtime``, which side registers the handler.
* ``job_states``: the statuses, the terminal and active sets and the allowed transitions of
  ``src/domain/job_states.py``, which the database trigger is generated from too.
* ``timing``: the lease, heartbeat and fence constants of ``src/domain/job_timing.py`` (defaults:
  the generator runs with a scrubbed environment).
* ``pause_reasons``, ``attempt_outcomes``, ``retry_outcomes``, ``advisory_locks``, ``wrapper``: the
  vocabularies and constants the wrapper and the periodic tasks use.

``runtime`` is ``rust`` for the kinds in :data:`RUST_RUNTIME_KINDS` and ``python`` for the rest. A
kind moves to Rust in the same change as its Rust handler, its routes and its ticks (plan 18,
section 2.5: the process that registers a handler for a kind runs it); a Rust test compares this
list with the handlers the runner registers.
"""

from __future__ import annotations

import json
from dataclasses import fields
from typing import Any


SCRIPT_REL = 'scripts/codegen/export_api_contracts.py'
SOURCES = [
    'src/config/job_kinds/',
    'src/domain/job_states.py',
    'src/domain/job_timing.py',
    'src/config/advisory_locks.py',
    'src/db/models/jobs.py',
    'src/db/repos/jobs.py',
    'src/tasks/runtime/job_task.py',
]

#: Kinds whose handler the Rust runner registers. Keep sorted.
RUST_RUNTIME_KINDS: tuple[str, ...] = ('noop', 'sleep_noop')


def _dump(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + '\n'


def render_job_kinds() -> dict[str, Any]:
    from src.config import advisory_locks
    from src.config.job_kinds import KINDS
    from src.config.job_kinds.kind import JobKind
    from src.config.queue import TaskQueues
    from src.db.models.jobs import ATTEMPT_OUTCOMES, PAUSE_REASONS
    from src.db.repos.jobs import RETRY_OUTCOMES
    from src.domain import job_states, job_timing
    from src.tasks.runtime import job_task

    unknown = sorted(set(RUST_RUNTIME_KINDS) - set(KINDS))
    if unknown:
        raise ValueError(f'RUST_RUNTIME_KINDS names kinds that do not exist: {unknown}')
    names = [f.name for f in fields(JobKind)]
    kinds = []
    for name in sorted(KINDS):
        kind = KINDS[name]
        row = {field: getattr(kind, field) for field in names}
        row['runtime'] = 'rust' if name in RUST_RUNTIME_KINDS else 'python'
        kinds.append(row)
    return {
        'generated_by': SCRIPT_REL,
        'sources': SOURCES,
        'kind_fields': [*names, 'runtime'],
        'kinds': kinds,
        'queues': list(TaskQueues.ALL),
        'job_states': {
            'statuses': list(job_states.STATUSES),
            'terminal': sorted(job_states.TERMINAL),
            'active': sorted(job_states.ACTIVE),
            'transitions': [list(pair) for pair in sorted(job_states.TRANSITIONS)],
        },
        'timing': {
            'lease_ttl_s': job_timing.LEASE_TTL_S,
            'heartbeat_tick_s': job_timing.HEARTBEAT_TICK_S,
            'fence_margin_s': job_timing.FENCE_MARGIN_S,
            'progress_min_interval_s': job_timing.PROGRESS_MIN_INTERVAL_S,
        },
        'pause_reasons': list(PAUSE_REASONS),
        'attempt_outcomes': list(ATTEMPT_OUTCOMES),
        'retry_outcomes': sorted(RETRY_OUTCOMES),
        'advisory_locks': {
            name: value
            for name, value in vars(advisory_locks).items()
            if name.isupper() and isinstance(value, int)
        },
        'wrapper': {
            'unavailable_project_statuses': sorted(job_task.UNAVAILABLE_PROJECT_STATUSES),
            'soft_limit_lead_s': job_task.SOFT_LIMIT_LEAD_S,
            'message_chars': job_task._MESSAGE_CHARS,
        },
    }


def render_job_kinds_json() -> str:
    return _dump(render_job_kinds())
