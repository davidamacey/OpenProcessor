"""A read that names another project's id through this project's path answers 404, never data.

The route sweep proves each scoped read reaches its handler with the bound project's own ids;
this file asks the same routes for the *other* project's ids (a job, near-duplicate run and group,
an item's region box) and requires the same 404 an unknown id gets, with none of the other
project's markers in the body.
"""

from __future__ import annotations

import types
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from curation.cross_project_leak._constants import SCOPED
from curation.cross_project_leak._seeded_files import own_image_path


if TYPE_CHECKING:
    from curation.cross_project_leak._transport import LeakEnv

pytestmark = pytest.mark.db

PAIRS = [('alpha', 'beta'), ('beta', 'alpha')]


def _routes(owner: str, ids: dict[str, str]) -> dict[str, tuple[str, dict[str, str]]]:
    return {
        f'/jobs/{ids["job_id"]}': ('job', {}),
        f'/near_duplicates/runs/{ids["run_id"]}': ('run', {}),
        f'/near_duplicates/groups/{ids["group_id"]}': ('group', {'level': 'image'}),
        f'/crops/{owner}-item-0004/region_thumbnail': ('thumb', {'box_id': 'b1'}),
    }


@pytest.mark.parametrize(('slug', 'other'), PAIRS)
def test_own_ids_answer_and_foreign_ids_are_not_found(
    leak_env: LeakEnv, slug: str, other: str
) -> None:
    client = TestClient(leak_env.app, raise_server_exceptions=False)
    base = SCOPED.format(project=slug)
    own = _routes(slug, leak_env.pg_ids[slug])
    foreign = _routes(other, leak_env.pg_ids[other])
    for (own_path, (_k, params)), (foreign_path, _) in zip(
        own.items(), foreign.items(), strict=True
    ):
        found = client.get(base + own_path, params=params)
        assert found.status_code == 200, f'{own_path}: {found.status_code} {found.text[:200]}'
        refused = client.get(base + foreign_path, params=params)
        assert refused.status_code == 404, (
            f"{slug} asked for {other}'s {foreign_path}: {refused.status_code} {refused.text[:200]}"
        )
        # the error may echo the requested id; it must carry nothing else of the other project
        body = refused.text.replace(foreign_path.rsplit('/', 1)[-1], '')
        assert f'{other}_' not in body, body


@pytest.mark.parametrize(('slug', 'other'), PAIRS)
def test_images_are_served_to_the_project_that_ingested_them_only(
    leak_env: LeakEnv, slug: str, other: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both projects ingest from one shared source root. Asking project ``slug`` for a file only
    ``other`` ingested is a 404 with no bytes, by absolute path, by relative path and through the
    named alias root (OWASP API1:2023: object-level authorization)."""
    from src.routers.curation import images as images_router

    aliases = {f'{name}-source': leak_env.root / 'images' / name for name in (slug, other)}
    monkeypatch.setattr(images_router, 'config', types.SimpleNamespace(source_path_aliases=aliases))
    client = TestClient(leak_env.app, raise_server_exceptions=False)
    base = SCOPED.format(project=slug)
    foreign = own_image_path(other)
    attempts = [
        ('/images/serve', foreign),
        ('/images/serve', f'{other}/{other}-lab-0001.jpg'),
        (f'/images/root/{other}-source', f'{other}-lab-0001.jpg'),
    ]
    own = [
        ('/images/serve', own_image_path(slug)),
        (f'/images/root/{slug}-source', f'{slug}-lab-0001.jpg'),
    ]
    for route, path in own:
        found = client.get(base + route, params={'path': path})
        assert found.status_code == 200, f'{route}: {found.status_code} {found.text[:200]}'
    for route, path in attempts:
        refused = client.get(base + route, params={'path': path})
        assert refused.status_code == 404, (
            f"{slug} fetched {other}'s image via {route}?path={path}: {refused.status_code}"
        )
        assert refused.headers['content-type'].startswith('application/json'), refused.headers
