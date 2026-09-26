"""The bound-project request context (see
``docs/design/openprocessor_internal/projects_plan.md`` §3.3).

A project is *always* bound while curation code runs -- there is no
``if project is None`` branch anywhere. Unbound access raises
:class:`ProjectNotBound` so a forgotten bind fails loudly instead of
silently falling back to ``default``.
"""

from __future__ import annotations

import contextvars
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from src.config.projects import ProjectRecord


@dataclass(frozen=True)
class BoundProject:
    """The project a request/task is currently scoped to."""

    record: ProjectRecord
    read_only: bool = False


class ProjectNotBound(RuntimeError):  # noqa: N818 - name fixed by the projects plan (§2.4/§3.3)
    """Curation code ran outside any bound project."""

    def __init__(self) -> None:
        super().__init__(
            'no project is bound in this context -- every curation request/task must bind a '
            'project before accessing project-scoped config or OpenSearch indexes'
        )


_BOUND: contextvars.ContextVar[BoundProject | None] = contextvars.ContextVar(
    'op_bound_project', default=None
)


def current_project() -> BoundProject:
    """The bound project for this context. Raises :class:`ProjectNotBound`
    when nothing has bound one -- there is deliberately no default
    fallback."""
    bound = _BOUND.get()
    if bound is None:
        raise ProjectNotBound
    return bound


def is_project_bound() -> bool:
    """Non-raising check, for call sites that need to branch on bound vs
    unbound *state itself* (for example a static test) rather than treat
    unbound as an error."""
    return _BOUND.get() is not None


@contextmanager
def bind_project(
    record: ProjectRecord, *, read_only: bool = False
) -> Generator[BoundProject, None, None]:
    """Bind ``record`` for the duration of the ``with`` block. Nesting is
    allowed and restores the outer binding (or unbound) on exit -- used by
    read-only cross-project reads such as ``clone_settings``'s source."""
    bound = BoundProject(record=record, read_only=read_only)
    token = _BOUND.set(bound)
    try:
        yield bound
    finally:
        _BOUND.reset(token)


def set_bound_project(record: ProjectRecord, *, read_only: bool = False) -> None:
    """Bind ``record`` for the rest of this task, with no reset.

    Only for FastAPI path dependencies: one request is one asyncio task,
    so there is no "later" scope to restore to, and resetting would be a
    no-op at best. Anything that must unbind afterward (scripts, cross-
    project reads) must use :func:`bind_project` instead.
    """
    _BOUND.set(BoundProject(record=record, read_only=read_only))


def project_api_base() -> str:
    """``{api_prefix}/projects/{bound slug}`` -- the base every served URL
    (and forward-looking route) is built from."""
    from src.config.curation import base_curation_config

    bound = current_project()
    return f'{base_curation_config().api_prefix}/projects/{bound.record.slug}'


async def run_in_executor_bound[T](
    loop: Any,
    executor: Any,
    fn: Callable[..., T],
    *args: Any,
) -> T:
    """``loop.run_in_executor`` does not copy the current ``contextvars``
    context into the worker thread, so a plain ``run_in_executor`` call
    silently loses the bound project (the callee then raises
    ``ProjectNotBound`` -- or worse, if it captured a stale global config
    at import time, silently touches the wrong project's data). Wrap the
    callable with the calling context so the binding survives the thread
    hop."""
    ctx = contextvars.copy_context()

    def _run_with_context() -> T:
        return ctx.run(fn, *args)

    return await loop.run_in_executor(executor, _run_with_context)


def project_env() -> dict[str, str]:
    """Env additions for a subprocess launch (``Popen(env=...)``) so a
    worker/trainer child process binds the same project as its parent."""
    return {'OP_PROJECT': current_project().record.slug}


def project_env_or_default() -> dict[str, str]:
    """Like :func:`project_env`, but falls back to ``OP_PROJECT``/``default``
    when nothing is bound -- for launch sites that run outside a request
    (for example a startup-time subprocess)."""
    if is_project_bound():
        return project_env()
    return {'OP_PROJECT': os.environ.get('OP_PROJECT', 'default')}
