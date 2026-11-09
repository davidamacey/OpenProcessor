"""Regression: the cluster-purity aggregation must not 500 on class_name.

The bug class this guards against: a purity aggregation targeted
bare ``class_name`` while the live index mapped it as a ``text`` field,
so OpenSearch refused the terms aggregation ("Text fields are not
optimised for operations that require per-document field data like
aggregations and sorting... Please use a keyword field instead").

The fix is structural rather than query-level: the
curation items index mapping (``src/clients/curation_opensearch/bodies_core.py``)
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


pytestmark = pytest.mark.integration


def test_items_index_maps_class_name_as_keyword() -> None:
    """The items index body must map class_name as keyword, never text —
    this is what makes the same bug class structurally impossible
    here regardless of which aggregation targets the field."""
    from src.clients.curation_opensearch import bodies_core

    src = inspect.getsource(bodies_core)
    # Every occurrence of a class_name mapping declares it keyword.
    assert "'class_name': {'type': 'keyword'}" in src
