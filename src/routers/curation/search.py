"""Semantic text search over curation items.

``GET /curation/search/text`` is the one pre-blessed exact path for
curation semantic search. No other ``/search/*`` prefix is used
anywhere in this module.

Gated end-to-end by ``OP_SEMANTIC_SEARCH_ENABLED`` (default off, inline
``os.getenv`` per house convention — see ``scores.py``'s
``scores_compute``/``select.py``'s diverse-overlay docstring for the
precedent this mirrors): disabled returns ``400`` with a message telling
the operator which flag to flip, same shape ``scores.py`` uses.
"""

from __future__ import annotations

import os
from typing import Annotated, Any

from fastapi import HTTPException, Query, Request

from src.routers.curation._common import OpenSearchDep, _ensure_indexes, router
from src.services.curation import semantic_search


def _semantic_search_enabled() -> bool:
    """Read fresh each call (not a module constant) — same
    ``OP_SCORES_ENABLED``/``OP_SELECT_DIVERSE_ENABLED`` live-read
    convention ``strategy_registry.py`` documents, so tests can
    ``monkeypatch.setenv`` without reimporting."""
    return os.environ.get('OP_SEMANTIC_SEARCH_ENABLED', '').strip().lower() in {
        '1',
        'true',
        'yes',
        'on',
    }


def _get_pe_encoder(request: Request) -> Any:
    """Same ``getattr(..., None)`` read ``health.py``'s ``/health/pe_text``
    uses against ``request.app.state.pe_encoder`` — populated by the
    ``src.main`` lifespan (non-fatal if the checkpoint never warmed; a
    still-cold or absent encoder here surfaces as a 503, not a crash)."""
    return getattr(request.app.state, 'pe_encoder', None)


@router.get('/search/text')
async def search_text(
    request: Request,
    opensearch: OpenSearchDep,
    q: Annotated[
        str, Query(min_length=1, description='Free-text query, e.g. "white pickup truck".')
    ],
    page: int = Query(1, ge=1),
    page_size: int = Query(30, ge=1, le=200),
    class_id: int | None = Query(None),
    cluster_id: int | None = Query(None),
    tab: str | None = Query(
        None,
        description=(
            'Optional review-tab cohort id (same set GET /curation/review/{tab} '
            'accepts) to scope the search — reuses '
            'review_queries.build_tab_query so results stay consistent '
            "with that tab's plain listing. Omitted searches the whole pool."
        ),
    ),
    date_from: str | None = Query(None, description='ISO date/datetime lower bound on created_at.'),
    date_to: str | None = Query(None, description='ISO date/datetime upper bound on created_at.'),
    max_rank: int | None = Query(None, ge=1),
    min_blur_ratio: float | None = Query(None, ge=0.0),
    hide_near_duplicates: bool = False,
    min_score: float | None = Query(
        None, ge=0.0, le=1.0, description='Drop hits below this cosine-similarity score.'
    ),
    include_test: bool = False,
) -> dict[str, Any]:
    """Semantic kNN search over the items index's ``pe_embedding`` field.

    Encodes ``q`` through the PE-Core-L14-336 text tower
    (:class:`src.clients.pe_encoder.PEEncoder`, off the event loop via the
    shared executor), builds a kNN query filtered the same way
    ``GET /curation/review/{tab}`` filters its cohorts, and returns
    ``{items, total, page, page_size}`` — same envelope shape as
    ``GET /curation/review/{tab}``.
    """
    if not _semantic_search_enabled():
        raise HTTPException(
            status_code=400,
            detail='semantic search is disabled (set OP_SEMANTIC_SEARCH_ENABLED=1 to enable)',
        )

    pe_encoder = _get_pe_encoder(request)
    if pe_encoder is None or not getattr(pe_encoder, 'text_ready', False):
        raise HTTPException(
            status_code=503,
            detail=(
                'PE text encoder not ready (checkpoint still warming, missing, '
                'or perception_models/torch not installed) — see GET /health/pe_text'
            ),
        )

    await _ensure_indexes(opensearch)

    # Lazy import: src.main imports the curation router package (which
    # imports this module) at module load time, so importing
    # get_shared_executor at *this* module's top level would be a circular
    # import. By call-time (a real HTTP request) src.main is fully loaded.
    from src.main import get_shared_executor

    try:
        executor = get_shared_executor()
    except RuntimeError:
        # Lifespan hasn't run (e.g. a router-only TestClient in tests) —
        # fall back to the event loop's default executor. encode_text is
        # still offloaded via run_in_executor either way; this only
        # changes which thread pool it runs on.
        executor = None

    try:
        return await semantic_search.semantic_text_search(
            opensearch=opensearch,
            pe_encoder=pe_encoder,
            executor=executor,
            query=q,
            page=page,
            page_size=page_size,
            class_id=class_id,
            cluster_id=cluster_id,
            tab=tab,
            date_from=date_from,
            date_to=date_to,
            max_rank=max_rank,
            min_blur_ratio=min_blur_ratio,
            hide_near_duplicates=hide_near_duplicates,
            min_score=min_score,
            include_test=include_test,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'semantic search failed: {exc}') from exc
