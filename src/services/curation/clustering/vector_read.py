"""The one reader of a ``knn_vector`` field over a query: every caller's vectors come through here.

Pages come from :func:`~src.services.curation.clustering.pit_pages.iter_pages` (sliced
point-in-time pages read concurrently), the vector is asked for through ``docvalue_fields`` and
decoded once per page by :func:`~src.services.curation.clustering.page_rows.hits_to_rows`.
Each caller keeps its own query and post-processing.

Per-box vectors (``region_box_embeddings``, a nested field) are not read here: the OpenSearch 3.7
notes and the k-NN pull request that added docvalue retrieval (k-NN #3321) do not document nested
``knn_vector`` fields, so that one shape reads ``_source`` through the same pager
(``region_box_rows.scroll_box_rows``).

Peak memory is two copies of the matrix while it is placed in id order (the page arrays, then the
sorted matrix, each freed as it is copied): 1,000,000 vectors of 1,024 dimensions are 4.1 GB each,
about 8.2 GB at the peak and 4.1 GB after it.
"""

from __future__ import annotations

from contextlib import aclosing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from src.services.curation.clustering.page_rows import hits_to_rows
from src.services.curation.clustering.pit_pages import DEFAULT_PAGE_SIZE, DEFAULT_SLICES, iter_pages


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Sequence

    from opensearchpy import AsyncOpenSearch


@dataclass(frozen=True)
class VectorPage:
    """The hits of one page that carry the vector: ids, float32 rows and the hits' ``_source``."""

    ids: list[str]
    matrix: np.ndarray
    sources: list[dict[str, Any]]


@dataclass(frozen=True)
class VectorRead:
    """Every vector of a query, sorted by id; ``truncated`` when ``max_rows`` was passed."""

    ids: list[str]
    matrix: np.ndarray
    truncated: bool = False


def empty_matrix() -> np.ndarray:
    return np.zeros((0, 0), dtype=np.float32)


async def iter_vector_pages(
    client: AsyncOpenSearch,
    *,
    index: str,
    field: str,
    query: dict[str, Any],
    source: Sequence[str] = (),
    n_slices: int = DEFAULT_SLICES,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> AsyncGenerator[VectorPage, None]:
    """Pages of the documents of ``query`` that have ``field``, in no fixed order."""
    pager = iter_pages(
        client,
        index=index,
        query=query,
        source=source,
        docvalue_fields=[field],
        n_slices=n_slices,
        page_size=page_size,
    )
    async with aclosing(pager) as pages:
        async for hits in pages:
            ids, matrix = hits_to_rows(hits, field)
            if matrix is None:
                continue
            have = set(ids)
            sources = [h.get('_source') or {} for h in hits if h['_id'] in have]
            yield VectorPage(ids, matrix, sources)


def _in_id_order(ids: list[str], pages: list[np.ndarray]) -> np.ndarray:
    """The pages as one matrix whose row ``k`` is the vector of the ``k``-th smallest id."""
    order = sorted(range(len(ids)), key=ids.__getitem__)
    rank = np.empty(len(order), dtype=np.intp)
    rank[order] = np.arange(len(order))
    out = np.empty((len(ids), pages[0].shape[1]), dtype=np.float32)
    start = 0
    while pages:
        page = pages.pop()
        stop = len(ids) - start
        out[rank[stop - len(page) : stop]] = page
        start += len(page)
    return out


async def read_vectors(
    client: AsyncOpenSearch,
    *,
    index: str,
    field: str,
    query: dict[str, Any],
    max_rows: int | None = None,
    on_page: Callable[[int], None] | None = None,
    n_slices: int = DEFAULT_SLICES,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> VectorRead:
    """``(ids, vectors)`` of the documents of ``query`` with ``field``, sorted by id.

    The fit and the scorers depend on the document set, not on page arrival order, so the result
    is always in id order. Past ``max_rows`` the read stops and returns the first ``max_rows``
    ids with an empty matrix and ``truncated``. ``on_page(n_rows_so_far)`` runs after each page
    (progress and cancel checks); an exception from it ends the read.
    """
    ids: list[str] = []
    pages: list[np.ndarray] = []
    pager = iter_vector_pages(
        client, index=index, field=field, query=query, n_slices=n_slices, page_size=page_size
    )
    async with aclosing(pager) as read:
        async for page in read:
            ids.extend(page.ids)
            pages.append(page.matrix)
            if on_page is not None:
                on_page(len(ids))
            if max_rows is not None and len(ids) > max_rows:
                return VectorRead(ids[:max_rows], empty_matrix(), truncated=True)
    if not ids:
        return VectorRead([], empty_matrix())
    return VectorRead(sorted(ids), _in_id_order(ids, pages))
