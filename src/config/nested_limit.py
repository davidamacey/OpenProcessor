"""The nested-document limit of one item document: the number and the box bound it gives.

OpenSearch k-NN issue 3623 (https://github.com/opensearch-project/k-NN/issues/3623; fixed by
PR 3624 for 3.10, not released; every 3.x is affected): on an index with derived source for
vectors, a parent document with 150 or more nested Lucene documents, counted across ALL its
nested fields, is read back with ANOTHER document's nested vectors, and any partial update,
scripted update, ``update_by_query`` or reindex then writes those wrong vectors to disk.

Measured on 3.6.0 with the real items index body: at 75 boxes per item (``region_boxes`` plus
``region_box_embeddings`` = 150 nested documents) 96% of the items returned wrong box vectors on
search, GET and mget; at 74 boxes (148) none did.

This module holds only the number and the arithmetic, so ``CurationConfig`` can validate against
it; the rule every writer applies is in :mod:`src.domain.nested_limit`. The structural fix (box
vectors leave the nested field) replaces both.
"""

from __future__ import annotations


NESTED_DOCS_LIMIT = 150
"""Nested Lucene documents of one parent at which k-NN issue 3623 corrupts vectors."""


def max_boxes_per_item(box_vector_lists: int) -> int:
    """The most boxes one item may hold when each box can carry one entry in each of
    ``box_vector_lists`` box-vector lists (one list: 74; two: 49; three: 37).

    Without assist attributes, which take further documents; the exact check is
    :func:`src.domain.nested_limit.nested_doc_count` at write time.
    """
    return (NESTED_DOCS_LIMIT - 1) // (1 + max(box_vector_lists, 0))


MAX_BOXES_PER_ITEM_DEFAULT_SPACE = max_boxes_per_item(1)
"""The bound with the one box space every project has (74): the ceiling of
``OP_REGION_MAX_BOXES_PER_WRITE`` and of ``max_regions_per_item``."""
