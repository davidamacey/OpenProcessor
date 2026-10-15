"""Project binding for script and worker entry points (projects_plan.md
§3.3: "every script entry point binds from ``--project`` (default
``default``) or ``OP_CURATION_PROJECT``").

A script binds its whole process (:func:`bind_process_project`), because
ContextVars do not follow it into the threads it starts. Every slug,
``default`` included, is resolved through the project registry, so the
stored status always applies (an archived project binds read-only; a
building/deleting/failed/deleted one refuses). A registry that cannot be
read refuses to bind at all: a script never guesses.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from src.config.env import env_str
from src.config.project_context import bind_process_project
from src.config.projects import DEFAULT_SLUG
from src.services.projects.registry import ProjectRegistry


if TYPE_CHECKING:
    import argparse

    from src.config.projects import ProjectRecord


# Statuses a script may act on: an archived project binds read-only (the
# guard refuses writes); building/deleting/deleted/failed refuse outright.
_BINDABLE = frozenset({'active', 'archived'})


def add_project_argument(parser: argparse.ArgumentParser) -> None:
    """``--project SLUG`` (default ``$OP_CURATION_PROJECT`` or ``default``)."""
    parser.add_argument(
        '--project',
        default=env_str('OP_CURATION_PROJECT', DEFAULT_SLUG),
        help='Project slug to act on (default: $OP_CURATION_PROJECT, else "default").',
    )


async def load_registry(opensearch_url: str | None = None) -> ProjectRegistry:
    """One read of the project registry through a short-lived guarded
    client. Raises ``SystemExit`` when it cannot be read."""
    from src.services.projects.guard import make_script_opensearch

    url = opensearch_url or _default_opensearch_url()
    client = make_script_opensearch([url])
    try:
        registry = ProjectRegistry(lambda: client)
        await registry.ensure_fresh()
    finally:
        await client.close()
    if not registry.refreshed:
        raise SystemExit(f'cannot read the project registry at {url}; refusing to guess a project')
    return registry


async def resolve_project(slug: str, *, opensearch_url: str | None = None) -> ProjectRecord:
    """The bindable record for ``slug``; raises ``SystemExit`` with a
    clear message for an unknown or unbindable project."""
    registry = await load_registry(opensearch_url)
    record = registry.get(slug)
    if record is None or record.status == 'deleted':
        raise SystemExit(f"no project named '{slug}'")
    if record.status not in _BINDABLE:
        raise SystemExit(f"project '{slug}' is {record.status}; refusing to run against it")
    return record


def bind_script_project(slug: str, *, opensearch_url: str | None = None) -> ProjectRecord:
    """Resolve ``slug`` and bind it for this whole script process. Call
    once, right after argument parsing and before anything reads
    project-scoped config. Must not run inside an event loop (it starts
    its own for the lookup); async entry points use
    :func:`abind_script_project`."""
    record = asyncio.run(resolve_project(slug, opensearch_url=opensearch_url))
    bind_process_project(record, read_only=record.status == 'archived')
    return record


async def abind_script_project(slug: str, *, opensearch_url: str | None = None) -> ProjectRecord:
    """:func:`bind_script_project` for an entry point already running in
    an event loop."""
    record = await resolve_project(slug, opensearch_url=opensearch_url)
    bind_process_project(record, read_only=record.status == 'archived')
    return record


def script_project_registry(
    opensearch_url: str | None = None, *, client: Any = None
) -> ProjectRegistry:
    """A project registry for a multi-project worker or maintenance script,
    reading the projects index through the script's own guarded client
    (``opensearch_url``, else ``$OPENSEARCH_URL``) rather than the API
    process's client factory.

    Pass ``client`` (the caller's own guarded client) to share it: the caller
    then closes the one client at shutdown. Without it the registry opens a
    client nothing can close, which is fine only for a short-lived script."""
    from src.services.projects.guard import make_script_opensearch

    shared = client or make_script_opensearch([opensearch_url or _default_opensearch_url()])
    return ProjectRegistry(lambda: shared)


def only_project(records: list[ProjectRecord], slug: str | None) -> list[ProjectRecord]:
    """``records`` restricted to ``slug`` (a worker's optional ``--project``
    filter); all of them when ``slug`` is ``None``."""
    return [r for r in records if slug is None or r.slug == slug]


def _default_opensearch_url() -> str:
    return env_str('OPENSEARCH_URL', 'http://opensearch:9200')
