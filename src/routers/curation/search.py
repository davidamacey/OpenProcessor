"""Semantic text search over curation items.

``GET /curation/search/text`` is the one pre-blessed exact path for
curation semantic search. No other ``/search/*`` prefix is used
anywhere in this module.

Gated end-to-end by ``OP_SEMANTIC_SEARCH_ENABLED`` (default on since #371; set
``0`` to turn it off, read with ``env_bool``): disabled returns ``400`` with a message telling
the operator which flag to flip, same shape ``scores.py`` uses.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import HTTPException, Query, Request

from src.config.env import env_bool
from src.routers.curation._common import OpenSearchDep, router
from src.routers.curation._item_filter_params import ItemFilterQuery  # noqa: TC001 - FastAPI
from src.routers.curation._page_models import SearchTextResponse
from src.services.curation import semantic_search
from src.services.curation.index_bootstrap import ensure_indexes
from src.services.curation.item_filter import item_filter_clauses


def _semantic_search_enabled() -> bool:
    """Read fresh each call (not a module constant) — same
    ``OP_SCORES_ENABLED``/``OP_SELECT_DIVERSE_ENABLED`` live-read
    convention ``strategy_registry.py`` documents, so tests can
    ``monkeypatch.setenv`` without reimporting."""
    return env_bool('OP_SEMANTIC_SEARCH_ENABLED', default=True)


def _get_pe_encoder(request: Request) -> Any:
    """Same ``getattr(..., None)`` read ``health.py``'s ``/health/pe_text``
    uses against ``request.app.state.pe_encoder`` — populated by the
    ``src.main`` lifespan (non-fatal if the checkpoint never warmed; a
    still-cold or absent encoder here surfaces as a 503, not a crash)."""
    return getattr(request.app.state, 'pe_encoder', None)


@router.get('/search/text', response_model=None, responses={200: {'model': SearchTextResponse}})
async def search_text(
    request: Request,
    opensearch: OpenSearchDep,
    item_filter: ItemFilterQuery,
    q: Annotated[
        str, Query(min_length=1, description='Free-text query, e.g. "white pickup truck".')
    ],
    page: int = Query(1, ge=1),
    page_size: int = Query(30, ge=1, le=200),
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
            detail='semantic search is disabled (OP_SEMANTIC_SEARCH_ENABLED is set off; unset it or set 1 to enable)',
        )

    try:
        item_filter_clauses(item_filter)  # a malformed band is the caller's 400, not an outage
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    pe_encoder = _get_pe_encoder(request)
    if pe_encoder is None or not getattr(pe_encoder, 'text_ready', False):
        raise HTTPException(
            status_code=503,
            detail=(
                'PE text encoder not ready (still warming, or no backend could load: '
                'no OP_PE_TEXT_ONNX_PATH file, no ready Triton pe_text_encoder, and '
                'perception_models/torch or the checkpoint unavailable) — see '
                'GET /health/pe_text'
            ),
        )

    await ensure_indexes(opensearch)

    # Looked up at call time so a test replaces the accessor in one place
    # (src.app.resources), not in every caller.
    from src.app.resources import get_shared_executor

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
            cluster_id=cluster_id,
            tab=tab,
            date_from=date_from,
            date_to=date_to,
            item_filter=item_filter,
            min_blur_ratio=min_blur_ratio,
            hide_near_duplicates=hide_near_duplicates,
            min_score=min_score,
            include_test=include_test,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'semantic search failed: {exc}') from exc
