"""Every consumer of "has a vector" agrees: a vectorless item is out, its embedded twin in.

The clause comes from ``embedding_state`` only; a second hand-written
``exists`` on the item embedding field is how consumers drift apart.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from curation.query_fakes import matches
from src.config.curation import ITEM_EMBEDDING_FIELD


REPO_ROOT = Path(__file__).resolve().parents[2]


def _proposal(crop_id: str, *, vector: bool) -> dict[str, Any]:
    doc: dict[str, Any] = {
        'crop_id': crop_id,
        'class_source': 'item_proposal',
        'class_validated': False,
        'test_holdout': False,
        'cluster_id': 7,
        'confidence': 0.9,
    }
    if vector:
        doc[ITEM_EMBEDDING_FIELD] = [0.1, 0.2]
    return doc


EMBEDDED_TWIN = _proposal('with', vector=True)
VECTORLESS = _proposal('without', vector=False)


def _bool(must: list[dict[str, Any]], must_not: list[dict[str, Any]]) -> dict[str, Any]:
    return {'bool': {'must': must, 'must_not': must_not}}


def _vlm_worker() -> dict[str, Any]:
    from scripts.curation.vlm_worker import _build_pending_query

    return _build_pending_query(0.8)


def _vlm_sweep() -> dict[str, Any]:
    from src.services.curation.autolabel.selection import vlm_selection_query

    return vlm_selection_query(class_id=None, cluster_id=None, classifier_confidence_skip_vlm=0.8)


def _vlm_cluster_scope() -> dict[str, Any]:
    from src.services.curation.autolabel.selection import vlm_selection_query

    return vlm_selection_query(class_id=None, cluster_id=7, classifier_confidence_skip_vlm=0.8)


def _review_all() -> dict[str, Any]:
    from src.services.curation.review_queries import build_tab_query

    must, must_not, _ = build_tab_query('all', include_test=True, text=None, max_rank=None)
    return _bool(must, must_not)


def _cluster_members() -> dict[str, Any]:
    from src.services.curation.clustering.cluster_geometry import _members_query

    return _members_query(7)


def _select_diverse() -> dict[str, Any]:
    from src.routers.curation.select import SelectDiverseScope, _build_scope_query

    return _build_scope_query(SelectDiverseScope())


@pytest.mark.parametrize(
    'build',
    [_vlm_worker, _vlm_sweep, _vlm_cluster_scope, _review_all, _cluster_members, _select_diverse],
    ids=lambda f: f.__name__,
)
def test_consumer_takes_the_embedded_twin_and_not_the_vectorless_item(build: Any) -> None:
    query = build()
    assert matches(EMBEDDED_TWIN, query)
    assert not matches(VECTORLESS, query)


_EXISTS_ON_ITEM_VECTOR = re.compile(
    r"""['"]exists['"]\s*:\s*\{\s*['"]field['"]\s*:\s*"""
    r"""(?:[\w.]*EMBEDDING_FIELD|['"]pe_embedding['"])"""
)


def test_nothing_outside_embedding_state_builds_its_own_exists_on_the_item_vector() -> None:
    offenders = []
    for root in ('src', 'scripts'):
        for path in sorted((REPO_ROOT / root).rglob('*.py')):
            if path.name == 'embedding_state.py':
                continue
            if _EXISTS_ON_ITEM_VECTOR.search(path.read_text()):
                offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == []
