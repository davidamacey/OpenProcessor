"""Per-tab query construction for ``GET /curation/review/{tab}``.

This module owns the entry point, :func:`build_tab_query`, which picks the tab's
clauses from :mod:`~src.services.curation.review_tab_clauses` (*what a tab matches*:
``must``/``must_not`` + the human-readable ``reason``) and applies the subject-size
limit. ``review_sorts.py`` owns *what order results come back in* — the two are
deliberately independent registries, so a tab's match logic can
never accidentally couple to its default sort.
"""

from __future__ import annotations

from typing import Any

from src.core.exceptions import RequestRejectedError
from src.services.curation import review_tab_clauses, review_tabs


def negative_frame_clause(on_negative_frame: bool) -> dict[str, Any]:
    """The one definition of the ``on_negative_frame`` filter, shared by
    ``GET /crops`` and the review queue."""
    marked: dict[str, Any] = {'term': {'on_negative_frame': True}}
    return marked if on_negative_frame else {'bool': {'must_not': marked}}


def build_tab_query(
    tab: str,
    *,
    include_test: bool,
    text: str | None,
    max_rank: int | None,
    region_status: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Return ``(must, must_not, reason)`` for one review tab.

    ``region_status`` (``regions`` tab only, ignored elsewhere): one of
    :data:`REGION_STATUS_FILTER_VALUES`. Raises ``RequestRejectedError(400, ...)``
    for an unrecognized ``tab`` or an unrecognized ``region_status`` — same
    behavior the review router had inline before this split.
    """
    if tab == 'regions':
        must, must_not, reason = review_tab_clauses.regions_tab(include_test, text, region_status)
    elif tab in review_tab_clauses.SIMPLE_TABS:
        must, must_not, reason = review_tab_clauses.SIMPLE_TABS[tab](include_test)
    else:
        raise RequestRejectedError(
            status_code=400,
            detail=f'unknown review tab: {tab}. Must be one of: {", ".join(review_tabs.KNOWN_TABS)}',
        )

    # Subject-size limit, applied once here for every tab that
    # serves it (it used to be honoured only by the two primary tabs).
    rank_limit = (
        max_rank
        if max_rank is not None
        else review_tabs.TAB_FILTER_DEFAULTS.get(tab, {}).get('max_rank')
    )
    if rank_limit is not None and 'max_rank' in review_tabs.tab_filters(tab):
        must.append({'range': {'crop_rank_in_image': {'lte': rank_limit}}})

    return must, must_not, reason
