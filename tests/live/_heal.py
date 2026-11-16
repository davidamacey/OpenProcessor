"""Heal and guard the live stack after a stack-acting test, without letting one step hide the rest.

The compose runner and the clock are injected so ``tests/test_live_heal.py`` runs it offline.
``subprocess.run(..., check=False)`` does not cover ``TimeoutExpired``: a stalled Docker daemon
raises it from the very first step, so every step here is tried on its own and its error kept.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import pytest


# The services every live test depends on (a stack-acting module names the extra ones it hurts).
STACK_SERVICES = ('api', 'jobs-worker', 'periodic-worker', 'jobs-beat')
GUARD_EXIT_CODE = 4
HEAL_BUDGET_S = 360.0
HEALTH_WAIT_S = 120.0

Run = Callable[..., Any]
States = Callable[[], Mapping[str, Mapping[str, Any]]]


class HealError(RuntimeError):
    """Every step of a heal that failed, one line each."""


def _bound() -> tuple[Run, States]:
    from tests.live import _compose

    return _compose.compose, _compose.service_states


def service_table(states: Mapping[str, Mapping[str, Any]], services: Iterable[str]) -> str:
    rows = []
    for name in services:
        row = states.get(name)
        if row is None:
            rows.append(f'{name}: no container')
        else:
            rows.append(
                f'{name}: State={row.get("State")} Health={row.get("Health") or "-"} '
                f'ExitCode={row.get("ExitCode")}'
            )
    return '\n'.join(rows)


def unhealthy(states: Mapping[str, Mapping[str, Any]], services: Iterable[str]) -> list[str]:
    bad = []
    for name in services:
        row = states.get(name)
        if (
            row is None
            or row.get('State') != 'running'
            or row.get('Health') not in (None, '', 'healthy')
        ):
            bad.append(name)
    return bad


def heal_stack(
    *,
    unpause: Iterable[str] = (),
    extra: Iterable[str] = (),
    run: Run | None = None,
    states: States | None = None,
    health_wait_s: float = HEALTH_WAIT_S,
    interval_s: float = 3.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Unpause, bring the stack up, wait until healthy; every step runs, errors are raised at the end."""
    bound_run, bound_states = _bound() if run is None or states is None else (run, states)
    run = run or bound_run
    states = states or bound_states
    services = (*STACK_SERVICES, *(s for s in extra if s not in STACK_SERVICES))
    errors: list[str] = []
    for service in unpause:
        try:
            run('unpause', service, check=False)
        except Exception as exc:
            errors.append(f'unpause {service}: {type(exc).__name__}: {exc}')
    try:
        run('up', '-d', '--no-deps', '--no-build', *services, timeout=240)
    except Exception as exc:
        errors.append(f'up {" ".join(services)}: {type(exc).__name__}: {exc}')
    # A thawed worker reports unhealthy until its next probe, so `up --wait` would refuse it.
    deadline = clock() + health_wait_s
    table = ''
    while True:
        try:
            current = states()
        except Exception as exc:
            errors.append(f'ps: {type(exc).__name__}: {exc}')
            break
        table = service_table(current, services)
        if not unhealthy(current, services):
            table = ''
            break
        if clock() >= deadline:
            errors.append(f'not healthy after {health_wait_s:.0f} s:\n{table}')
            break
        sleep(interval_s)
    if errors:
        raise HealError('\n'.join(errors))


def require_healthy_stack(
    *,
    unpause: Iterable[str] = (),
    extra: Iterable[str] = (),
    run: Run | None = None,
    states: States | None = None,
    budget_s: float = HEAL_BUDGET_S,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    where: str = '',
) -> None:
    """Pass when the stack is healthy; else heal once within ``budget_s``, else end the session (code 4)."""
    bound_run, bound_states = _bound() if run is None or states is None else (run, states)
    run = run or bound_run
    states = states or bound_states
    services = (*STACK_SERVICES, *(s for s in extra if s not in STACK_SERVICES))
    problem = ''
    try:
        if not unhealthy(states(), services):
            return
    except Exception as exc:
        problem = f'ps failed: {type(exc).__name__}: {exc}'
    started = clock()
    try:
        heal_stack(
            unpause=unpause,
            extra=extra,
            run=run,
            states=states,
            health_wait_s=max(1.0, min(HEALTH_WAIT_S, budget_s)),
            clock=clock,
            sleep=sleep,
        )
        return
    except HealError as exc:
        problem = f'{problem}\n{exc}'.strip()
    try:
        table = service_table(states(), services)
    except Exception as exc:
        table = f'ps failed: {type(exc).__name__}: {exc}'
    message = (
        f'live stack cannot be healed {where} (after {clock() - started:.0f} s of '
        f'{budget_s:.0f} s). Check the Docker daemon first, then the services named below.\n'
        f'{table}\n{problem}'
    )
    if os.environ.get('PYTEST_XDIST_WORKER'):
        # No documented effect of pytest.exit in a worker; a stack-acting module should not run here.
        pytest.fail(message, pytrace=False)
    pytest.exit(message, returncode=GUARD_EXIT_CODE)
