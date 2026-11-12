"""run_over_projects: one project when --project is given, else every active + archived project
under its own binding (archived read-only), with the worst exit code returned."""

from __future__ import annotations

import argparse
import contextlib
import types

import pytest

from scripts.curation import _prune_cli


class _Registry:
    async def refresh_strict(self) -> None:
        return None

    def active_projects(self) -> list[types.SimpleNamespace]:
        return [types.SimpleNamespace(slug='a', status='active')]

    def archived_projects(self) -> list[types.SimpleNamespace]:
        return [types.SimpleNamespace(slug='b', status='archived')]


@pytest.mark.asyncio
async def test_all_projects_run_each_under_its_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    bound: list[tuple[str, bool]] = []

    @contextlib.contextmanager
    def _bind(record: types.SimpleNamespace, *, read_only: bool):
        bound.append((record.slug, read_only))
        yield

    monkeypatch.setattr(_prune_cli, 'script_project_registry', lambda: _Registry())
    monkeypatch.setattr(_prune_cli, 'bind_project', _bind)
    seen: list[int] = []

    def _one(_args: argparse.Namespace) -> int:
        seen.append(len(bound))
        return 3 if len(bound) == 1 else 0

    rc = await _prune_cli.run_over_projects(argparse.Namespace(project=None), _one)
    assert bound == [('a', False), ('b', True)]
    assert seen == [1, 2]
    assert rc == 3


@pytest.mark.asyncio
async def test_one_project_binds_by_slug_only(monkeypatch: pytest.MonkeyPatch) -> None:
    slugs: list[str] = []

    async def _abind(slug: str) -> None:
        slugs.append(slug)

    monkeypatch.setattr(_prune_cli, 'abind_script_project', _abind)
    rc = await _prune_cli.run_over_projects(argparse.Namespace(project='cars'), lambda _a: 7)
    assert slugs == ['cars']
    assert rc == 7


def test_parser_defaults_to_dry_run_over_every_project() -> None:
    p = _prune_cli.new_prune_parser(None)
    ns = p.parse_args([])
    assert (ns.apply, ns.project) == (False, None)
    assert p.parse_args(['--apply', '--project', 'x']).apply is True
