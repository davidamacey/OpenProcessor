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


def _routes(owner: str, ids: dict[str, str]) -> dict[str, dict[str, str]]:
    """Every read that names an id, with its query; the same set, in the same order, for any
    project's ids."""
    loop, run = ids['loop_id'], ids['assist_run_id']
    train, bakeoff = ids['train_job_id'], ids['bakeoff_job_id']
    imp, bulk = ids['import_id'], ids['bulk_run_id']
    return {
        f'/jobs/{ids["job_id"]}': {},
        f'/near_duplicates/runs/{ids["run_id"]}': {},
        f'/near_duplicates/groups/{ids["group_id"]}': {'level': 'image'},
        f'/crops/{owner}-item-0004/region_thumbnail': {'box_id': 'b1'},
        f'/assist/loops/{loop}': {},
        f'/assist/loops/{loop}/nodes': {},
        f'/assist/runs/{run}': {},
        f'/bakeoff/status/{bakeoff}': {},
        f'/bakeoff/matrix/{bakeoff}': {},
        f'/bakeoff/results/{bakeoff}': {},
        f'/clusters/partitions/{ids["partition_id"]}': {},
        f'/datasets/imports/{imp}': {},
        f'/datasets/imports/{imp}/entries': {},
        f'/datasets/imports/{imp}/issues': {},
        f'/ingest/bulk/{bulk}': {},
        f'/ingest/bulk/{bulk}/failures': {},
        f'/ingest/sources/{ids["source_id"]}': {},
        f'/pipeline/auto_label/status/{ids["autolabel_job_id"]}': {},
        f'/reprocess/jobs/{ids["reprocess_job_id"]}': {},
        f'/select/diverse/{ids["selection_job_id"]}/items': {},
        f'/train/status/{train}': {},
        f'/train/manifest/{train}': {},
        f'/train/artifacts/{train}/results.csv': {},
        f'/train/promote/{train}/jobs/{ids["promote_id"]}': {},
    }


# A run's log tail is an empty list for a run the project does not have, not a 404.
LOG_TAIL = '/train/log/tail/{}'


@pytest.mark.parametrize(('slug', 'other'), PAIRS)
def test_own_ids_answer_and_foreign_ids_are_not_found(
    leak_env: LeakEnv, slug: str, other: str
) -> None:
    client = TestClient(leak_env.app, raise_server_exceptions=False)
    base = SCOPED.format(project=slug)
    own = _routes(slug, leak_env.pg_ids[slug])
    foreign = _routes(other, leak_env.pg_ids[other])
    for (own_path, params), (foreign_path, _) in zip(own.items(), foreign.items(), strict=True):
        found = client.get(base + own_path, params=params)
        assert found.status_code == 200, f'{own_path}: {found.status_code} {found.text[:200]}'
        refused = client.get(base + foreign_path, params=params)
        assert refused.status_code == 404, (
            f"{slug} asked for {other}'s {foreign_path}: {refused.status_code} {refused.text[:200]}"
        )
        # the error may echo the requested id; it must carry nothing else of the other project
        text = refused.text
        for value in sorted(
            {*foreign_path.split('/'), *leak_env.pg_ids[other].values()}, key=len, reverse=True
        ):
            text = text.replace(value, '') if value else text
        assert f'{other}_' not in text, text
        assert f'{other}-' not in text, text


@pytest.mark.parametrize(('slug', 'other'), PAIRS)
def test_a_run_log_tail_names_only_the_projects_own_run(
    leak_env: LeakEnv, slug: str, other: str
) -> None:
    client = TestClient(leak_env.app, raise_server_exceptions=False)
    base = SCOPED.format(project=slug)
    own = client.get(base + LOG_TAIL.format(leak_env.pg_ids[slug]['train_job_id']))
    assert own.status_code == 200
    assert own.json()['lines'] == [f'{slug}-epoch 1', f'{slug}-epoch 2']
    foreign = client.get(base + LOG_TAIL.format(leak_env.pg_ids[other]['train_job_id']))
    assert foreign.status_code == 200
    assert foreign.json()['lines'] == []


@pytest.mark.parametrize(('slug', 'other'), PAIRS)
def test_images_are_served_to_the_project_that_ingested_them_only(
    leak_env: LeakEnv, slug: str, other: str
) -> None:
    """Asking project ``slug`` for an image only ``other`` ingested, by id, is a 404 with no bytes
    on the file, the thumbnail and the detail route, the same answer an unknown id gets (OWASP
    API1:2023: object-level authorization). Its own image is served."""
    client = TestClient(leak_env.app, raise_server_exceptions=False)
    base = SCOPED.format(project=slug)
    for suffix in ('/file', '/thumbnail', ''):
        found = client.get(f'{base}/images/{slug}-img-0002{suffix}')
        assert found.status_code == 200, f'{suffix!r}: {found.status_code} {found.text[:200]}'
        refused = client.get(f'{base}/images/{other}-img-0002{suffix}')
        unknown = client.get(f'{base}/images/no-such-image{suffix}')
        assert refused.status_code == unknown.status_code == 404, (
            f"{slug} read {other}'s image {suffix!r}: {refused.status_code}"
        )
        assert refused.headers['content-type'].startswith('application/json'), refused.headers
        # The 404 only echoes the id asked for: the same body an unknown id gets.
        assert refused.json()['detail'] == unknown.json()['detail'].replace(
            'no-such-image', f'{other}-img-0002'
        )
