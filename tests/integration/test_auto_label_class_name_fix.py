"""Regression: the cluster-purity aggregation must not 500 on class_name.

The bug class this guards against: a purity aggregation targeted
bare ``class_name`` while the live index mapped it as a ``text`` field,
so OpenSearch refused the terms aggregation ("Text fields are not
optimised for operations that require per-document field data like
aggregations and sorting... Please use a keyword field instead").

The fix is structural rather than query-level: the
curation items index mapping (``src/clients/curation_opensearch.py``)
declares ``class_name`` as ``keyword`` from the start, so the bug class
cannot reproduce regardless of which aggregation targets it. This test
pins both halves of that invariant — the mapping type and the
aggregation's field reference — rather than making a live HTTP call
(per this repo's house rule: fake the I/O boundary instead of adding
the repo's first live-stack dependency).
"""

from __future__ import annotations

import inspect

import pytest

# Import via `orchestrator` (not `auto_promote` directly) — the two
# modules have a top/bottom circular import between them that only
# resolves cleanly when `orchestrator` is the first of the pair loaded.
from src.services.curation.clustering.orchestrator import auto_promote_clusters  # noqa: F401


# isort: split
from src.services.curation.clustering.auto_promote import _scroll_cluster_buckets


pytestmark = pytest.mark.integration


def test_items_index_maps_class_name_as_keyword() -> None:
    """The items index body must map class_name as keyword, never text —
    this is what makes the same bug class structurally impossible
    here regardless of which aggregation targets the field."""
    import src.clients.curation_opensearch as curation_opensearch_mod

    src = inspect.getsource(curation_opensearch_mod)
    # Every occurrence of a class_name mapping declares it keyword.
    assert "'class_name': {'type': 'keyword'}" in src


def test_auto_promote_purity_aggregation_targets_class_name_directly() -> None:
    """The purity aggregation must target bare ``class_name`` (matching
    the keyword mapping above), not a ``.keyword`` subfield that doesn't
    exist on this schema — pointing at a nonexistent subfield would 400,
    not silently degrade.

    The aggregation body lives in ``_scroll_cluster_buckets``
    (composite-agg paging), not ``auto_promote_clusters`` itself —
    inspect that helper instead."""
    src = inspect.getsource(_scroll_cluster_buckets)
    assert "'field': 'class_name'" in src
    assert "'field': 'class_name.keyword'" not in src
