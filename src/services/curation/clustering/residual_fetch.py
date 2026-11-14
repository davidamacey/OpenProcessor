"""Residual-pool embedding fetch for the item scorers and the UMAP views.

Pulls embeddings for residual crops (no confident class assigned) from the items index,
L2-normalized, from the vector field the caller names (the project's primary item space,
``embedding_space(Target.ITEM).vector_field``). One read, :func:`fetch_residual_embeddings`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from opensearchpy.exceptions import OpenSearchException

from src.config.curation_accessors import items_index
from src.core.logging import get_logger
from src.domain.embedding_state import embedded_clause
from src.domain.ingest_class_sources import confident_class_sources
from src.services.curation.clustering.pit_pages import DEFAULT_PAGE_SIZE, DEFAULT_SLICES
from src.services.curation.clustering.vector_read import empty_matrix, read_vectors


if TYPE_CHECKING:
    from collections.abc import Callable

    from opensearchpy import AsyncOpenSearch


logger = get_logger(__name__)


# Crops with these class_source values are *confidently* labeled and
# must NEVER be pulled into the residual pool.
#
# Specifically (src.domain.ingest_class_sources, derived from
# the configured ingest profiles -- resolved once at import like other
# OP_* config):
#   {secondary}_model — the ingest classifier labeled it above its floor
#                       ({primary}_model too when the primary assigns_class)
#   vlm               — the VLM matched a registered class name
#   human             — human-validated label
#
# Everything else (low-conf / unlabeled proposals, vlm_unmatched,
# vlm_new_class_pending, null) is residual and goes into the cluster.
CONFIDENT_CLASS_SOURCES = confident_class_sources()


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    matrix /= norms
    return matrix


def residual_pool_query(*, field: str, extra_must: list[dict[str, Any]] | None) -> dict[str, Any]:
    """The residual-pool gate shared by both fetch variants.

    Embedded crops that are not validated, not confidently labeled (see
    ``CONFIDENT_CLASS_SOURCES``) and not excluded; ``extra_must`` narrows it further."""
    filt: list[dict[str, Any]] = [embedded_clause(field), *(extra_must or ())]
    must_not: list[dict[str, Any]] = [
        {'term': {'class_validated': True}},
        {'terms': {'class_source': list(CONFIDENT_CLASS_SOURCES)}},
        # Human-ignored crops (blurry / unidentifiable) stay out of the
        # residual pool so they're never re-clustered.
        {'term': {'class_excluded': True}},
    ]
    return {'bool': {'filter': filt, 'must_not': must_not}}


async def _precount(client: AsyncOpenSearch, query: dict[str, Any], progress: Any | None) -> int:
    """Pre-count so the progress bar has a real total (0 without progress or on failure).

    A separate count query is cheap (no _source, no scroll) and lets the
    dashboard show 12,000 / 90,000 instead of "working...".
    """
    total_estimate = 0
    if progress is not None:
        try:
            count_resp = await client.count(index=items_index(), body={'query': query})
            total_estimate = int(count_resp.get('count', 0))
            progress.update(processed=0, total=total_estimate)
        except OpenSearchException as exc:
            logger.debug('curation_embedding_count_failed', error=str(exc))
    return total_estimate


def _progress_hook(progress: Any | None, total_estimate: int) -> Callable[[int], None] | None:
    if progress is None:
        return None

    def on_page(n_rows: int) -> None:
        progress.update(processed=n_rows, total=total_estimate or n_rows)
        progress.raise_if_cancelled()

    return on_page


async def fetch_residual_embeddings(
    client: AsyncOpenSearch,
    *,
    field: str,
    extra_must: list[dict[str, Any]] | None = None,
    n_slices: int = DEFAULT_SLICES,
    page_size: int = DEFAULT_PAGE_SIZE,
    progress: Any | None = None,
) -> tuple[list[str], np.ndarray]:
    """Unvalidated crops that have a vector in ``field`` (the residual pool), by document id.

    One read (:func:`~src.services.curation.clustering.vector_read.read_vectors`: sliced
    point-in-time pages read at once, the vector through ``docvalue_fields``), rows L2-normalised.
    When ``progress`` is supplied it is updated after each page and its cancel flag is checked,
    so an operator's cancel takes effect within one page.
    """
    query = residual_pool_query(field=field, extra_must=extra_must)
    total_estimate = await _precount(client, query, progress)

    read = await read_vectors(
        client,
        index=items_index(),
        field=field,
        query=query,
        on_page=_progress_hook(progress, total_estimate),
        n_slices=n_slices,
        page_size=page_size,
    )
    if not read.ids:
        return [], empty_matrix()
    return read.ids, _normalize_rows(read.matrix)
