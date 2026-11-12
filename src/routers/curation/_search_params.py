"""What the three semantic search routes share: the filter query parameters, the
enabled check, the filter validation and the keyword arguments every search service takes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import Depends, HTTPException, Query

from src.config.env import env_bool
from src.services.curation.image_join import resolve_filter_join
from src.services.curation.item_filter import ItemFilter, item_filter_clauses
from src.services.embedding_spaces.readers import item_write_fields


def semantic_search_enabled() -> bool:
    """Read fresh each call (not a module constant) -- the live-read convention
    ``strategy_registry.py`` documents, so tests can ``monkeypatch.setenv`` without
    reimporting."""
    return env_bool('OP_SEMANTIC_SEARCH_ENABLED', default=True)


@dataclass
class SearchWindow:
    """The page, filters and space selection of a semantic search request."""

    page: Annotated[int, Query(ge=1)] = 1
    page_size: Annotated[int, Query(ge=1, le=200)] = 30
    date_from: Annotated[
        str | None, Query(description='ISO date/datetime lower bound on created_at.')
    ] = None
    date_to: Annotated[
        str | None, Query(description='ISO date/datetime upper bound on created_at.')
    ] = None
    min_blur_ratio: Annotated[float | None, Query(ge=0.0)] = None
    hide_near_duplicates: bool = False
    min_score: Annotated[
        float | None,
        Query(ge=0.0, le=1.0, description='Drop hits below this cosine-similarity score.'),
    ] = None
    include_test: bool = False
    space: Annotated[
        str | None,
        Query(
            description=(
                "The space to search (a live space key of the target); default the target's "
                'primary. Echoed in the response as `space`.'
            )
        ),
    ] = None
    fuse: Annotated[
        list[str] | None,
        Query(
            description=(
                'More live space keys to search; the per-space rankings are merged by reciprocal '
                'rank fusion. Each item keeps the cosine `semantic_score` of the first space that '
                'found it and names it in `score_space`.'
            )
        ),
    ] = None

    def service_kwargs(self) -> dict[str, Any]:
        """The keyword arguments every ``semantic_*_search`` service takes from this window."""
        return {
            'page': self.page,
            'page_size': self.page_size,
            'date_from': self.date_from,
            'date_to': self.date_to,
            'min_blur_ratio': self.min_blur_ratio,
            'hide_near_duplicates': self.hide_near_duplicates,
            'min_score': self.min_score,
            'include_test': self.include_test,
            'space_key': self.space,
            'fuse': self.fuse or [],
        }


SearchWindowQuery = Annotated[SearchWindow, Depends()]


async def checked_filter_join(opensearch: Any, item_filter: ItemFilter) -> Any:
    """Refuse (400) when semantic search is off or the filter is malformed; else the image join."""
    if not semantic_search_enabled():
        raise HTTPException(
            status_code=400,
            detail='semantic search is disabled (OP_SEMANTIC_SEARCH_ENABLED is set off; unset it or set 1 to enable)',
        )
    try:
        join = await resolve_filter_join(opensearch, item_filter)
        item_filter_clauses(
            item_filter, embedding=await item_write_fields(), image_join=join
        )  # a malformed band is the caller's 400, not an outage
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return join
