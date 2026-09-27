"""An in-memory project registry for worker tests.

Multi-project workers discover their fleet through
``src.services.projects.script_binding.script_project_registry``, which
reads the projects index over the network. Worker tests that fake
OpenSearch at ``make_script_opensearch`` must fake this too, or the
registry's refresh hits the test's deliberately unresolvable host and
the worker sees no projects to poll.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import pytest

    from src.config.projects import ProjectRecord


class StaticProjectRegistry:
    """The :class:`~src.services.projects.registry.ProjectRegistry` read
    surface over a fixed list of records; never does I/O."""

    refreshed = True

    def __init__(self, records: list[ProjectRecord]) -> None:
        self._by_slug = {r.slug: r for r in records}

    async def ensure_fresh(self) -> None:
        return None

    def snapshot(self) -> dict[str, ProjectRecord]:
        return dict(self._by_slug)

    def get(self, slug: str) -> ProjectRecord | None:
        return self._by_slug.get(slug)

    def active_projects(self) -> list[ProjectRecord]:
        return [r for r in self._by_slug.values() if r.status == 'active']

    def archived_projects(self) -> list[ProjectRecord]:
        return [r for r in self._by_slug.values() if r.status == 'archived']


def default_project_record() -> ProjectRecord:
    """``default`` as bootstrap creates it from the current env."""
    from src.config.curation import base_curation_config
    from src.config.projects import DEFAULT_SLUG, new_project_record

    return new_project_record(DEFAULT_SLUG, base_curation_config())


def install_static_project_registry(
    monkeypatch: pytest.MonkeyPatch, records: list[ProjectRecord] | None = None
) -> StaticProjectRegistry:
    """Make every ``script_project_registry(...)`` call return a
    :class:`StaticProjectRegistry` over ``records`` (default: just
    ``default``)."""
    registry = StaticProjectRegistry(records if records is not None else [default_project_record()])

    def _factory(*_a: Any, **_kw: Any) -> StaticProjectRegistry:
        return registry

    monkeypatch.setattr('src.services.projects.script_binding.script_project_registry', _factory)
    return registry
