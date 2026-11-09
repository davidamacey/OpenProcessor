"""A read that names another project's id through this project's path answers 404, never data.

The route sweep proves each scoped read reaches its handler with the bound project's own ids;
this file asks the same routes for the *other* project's ids (a job, near-duplicate run and group,
an item's region box) and requires the same 404 an unknown id gets, with none of the other
project's markers in the body.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from curation.cross_project_leak._constants import SCOPED


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
