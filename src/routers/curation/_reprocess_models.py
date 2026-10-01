"""Wire models of the reprocess routes (W10.14). The request/response
shapes live in the service layer
(:mod:`src.services.curation.reprocess_models`) because the same models are
the persisted job request and ``ActivationImpact.suggested_reprocess``;
this module re-exports them and adds the one wire-only field, the post-write
items the single-target routes return."""

from __future__ import annotations

from pydantic import Field

from src.routers.curation._item_models import ItemDoc  # noqa: TC001 - pydantic field type
from src.services.curation.reprocess_models import (
    BreakdownRow,
    ReprocessFilter,
    ReprocessJobInfo,
    ReprocessOneRequest,
    ReprocessRequest,
    ReprocessResponse,
    ReprocessScopeResult,
    ReprocessTargets,
)


class ReprocessWireResponse(ReprocessResponse):
    """``ReprocessResponse`` plus the post-write ``items`` (only the
    single image / crop routes fill it; the GUI adopts them)."""

    items: list[ItemDoc] = Field(default_factory=list)


__all__ = [
    'BreakdownRow',
    'ReprocessFilter',
    'ReprocessJobInfo',
    'ReprocessOneRequest',
    'ReprocessRequest',
    'ReprocessScopeResult',
    'ReprocessTargets',
    'ReprocessWireResponse',
]
