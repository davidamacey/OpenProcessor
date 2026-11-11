"""``GET {P}/clusters``: cards, review counts and representatives come from the bound project only."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from curation import query_fakes
from curation.cross_project_leak._constants import CLASS_NAMES, SCOPED, SLUGS, cluster_uid
from src.services.curation import review_tab_clauses


if TYPE_CHECKING:
    from curation.cross_project_leak._transport import LeakEnv


pytestmark = pytest.mark.db


@pytest.fixture(autouse=True)
def _script_twins(monkeypatch: pytest.MonkeyPatch) -> None:
    # The fake cannot run painless: without a Python twin the whole aggregation is dropped.
    twins = query_fakes.SCRIPT_EVALUATORS
    monkeypatch.setitem(twins, review_tab_clauses.DETECTOR_DISAGREES_SCRIPT, lambda _doc: False)


def _only_the_bound_project(leak_env: LeakEnv, slug: str, body: dict) -> None:
    """The response names no other project's data and OpenSearch saw only the bound project's indexes."""
    text = json.dumps(body)
    for other in (s for s in SLUGS if s != slug):
        assert CLASS_NAMES[other] not in text
        assert f'{other}-item' not in text
    own = {str(i) for i in leak_env.records[slug].resources.indexes.values()}
    # The deployment-wide config index is shared by design (the project guard allows it).
    touched = {idx for _b, _m, _u, idx in leak_env.accesses if idx} - {'op_global_configs'}
    assert touched, 'the route reached OpenSearch'
    assert touched <= own, f'touched foreign indexes: {touched - own}'
    assert {bound for bound, *_ in leak_env.accesses} == {slug}


@pytest.mark.parametrize('slug', SLUGS)
def test_cluster_list_serves_only_the_bound_projects_data(leak_env: LeakEnv, slug: str) -> None:
    client = TestClient(leak_env.app, raise_server_exceptions=False)
    leak_env.accesses.clear()

    resp = client.get(
        f'{SCOPED.format(project=slug)}/clusters',
        params={'per_cluster': 2, 'sort': 'needs_review_desc'},
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['items'], 'the seeded project has clustered items'
    assert body['truncated'] is False
    assert all(card['review_counts'] for card in body['items'])
    assert f'{slug}-item' in json.dumps(body)  # representatives are this project's crops
    _only_the_bound_project(leak_env, slug, body)


@pytest.mark.parametrize('slug', SLUGS)
def test_cluster_facets_serve_only_the_bound_projects_data(leak_env: LeakEnv, slug: str) -> None:
    client = TestClient(leak_env.app, raise_server_exceptions=False)
    leak_env.accesses.clear()

    resp = client.get(f'{SCOPED.format(project=slug)}/clusters/{cluster_uid(slug)}/facets')

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['total'] > 0, 'the seeded project has members in the seeded cluster'
    _only_the_bound_project(leak_env, slug, body)


@pytest.mark.parametrize('slug', SLUGS)
def test_cluster_card_serves_only_the_bound_projects_data(leak_env: LeakEnv, slug: str) -> None:
    client = TestClient(leak_env.app, raise_server_exceptions=False)
    leak_env.accesses.clear()

    resp = client.get(f'{SCOPED.format(project=slug)}/clusters/{cluster_uid(slug)}')

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['cluster_uid'] == cluster_uid(slug)
    assert body['size'] > 0
    assert f'{slug}-item' in json.dumps(body)  # representatives are this project's crops
    _only_the_bound_project(leak_env, slug, body)


def test_a_foreign_projects_cluster_is_not_found(leak_env: LeakEnv) -> None:
    client = TestClient(leak_env.app, raise_server_exceptions=False)

    resp = client.get(f'{SCOPED.format(project="alpha")}/clusters/{cluster_uid("beta")}')

    assert resp.status_code == 404
    assert resp.json()['detail']['error'] == 'cluster_not_found'
