"""``GET {P}/clusters``: cards, review counts and representatives come from the bound project only."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from curation import query_fakes
from curation.cross_project_leak._constants import CLASS_NAMES, SCOPED, SLUGS
from src.services.curation import cluster_cards, review_tab_clauses


if TYPE_CHECKING:
    from curation.cross_project_leak._transport import LeakEnv


@pytest.fixture(autouse=True)
def _script_twins(monkeypatch: pytest.MonkeyPatch) -> None:
    # The fake cannot run painless: without a Python twin the whole aggregation is dropped.
    twins = query_fakes.SCRIPT_EVALUATORS
    monkeypatch.setitem(twins, cluster_cards.MEASURED_SCRIPT, lambda _doc: False)
    monkeypatch.setitem(twins, cluster_cards.FITS_SCRIPT, lambda _doc: False)
    monkeypatch.setitem(twins, review_tab_clauses.DETECTOR_DISAGREES_SCRIPT, lambda _doc: False)


@pytest.mark.parametrize('slug', SLUGS)
def test_cluster_list_serves_only_the_bound_projects_data(leak_env: LeakEnv, slug: str) -> None:
    client = TestClient(leak_env.app, raise_server_exceptions=False)
    others = [s for s in SLUGS if s != slug]
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
    text = json.dumps(body)
    assert f'{slug}-item' in text  # representatives are this project's crops
    for other in others:
        assert f'{other}-item' not in text
        assert CLASS_NAMES[other] not in text
    own = {str(i) for i in leak_env.records[slug].resources.indexes.values()}
    # The deployment-wide config index is shared by design (the project guard allows it).
    touched = {idx for bound, _m, _u, idx in leak_env.accesses if idx} - {'op_global_configs'}
    assert touched, 'the route reached OpenSearch'
    assert touched <= own, f'touched foreign indexes: {touched - own}'
    assert {bound for bound, *_ in leak_env.accesses} == {slug}
