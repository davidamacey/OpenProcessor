"""Wire models for ``GET /curation/methods``.

Documentation/OpenAPI models only — the handler still returns
``strategy_registry.get_registry``'s plain dict verbatim. This module exists
so the generated TS contract declares the fields every consumer already
reads off a real response (``field_coverage`` / ``field_coverage_total`` in
particular), instead of typing the whole payload as a free-form object.

``StrategyEntry`` uses ``extra='allow'``: different axes attach different
extras (``purity``, ``requires_banner``, ``version``, ``writes``, …) that
this model does not enumerate — see ``strategy_registry.py`` for the axis
builders that produce them.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class StrategyEntry(BaseModel):
    """One strategy/registry row across any axis (cluster / score / sort /
    overlay / export / detection_profile / prompt_pack)."""

    model_config = ConfigDict(extra='allow')

    id: str
    axis: str
    label: str
    status: Literal['stable', 'experimental', 'shadow', 'disabled']
    default: bool
    settable: bool | None = None
    requires_field: str | None = None
    field_coverage: int | None = None
    field_coverage_total: int | None = None


class MethodsResponse(BaseModel):
    """``GET /curation/methods`` response: every strategy across every axis
    plus the feature flags that gated each entry's status."""

    strategies: list[StrategyEntry]
    flags: dict[str, bool]


__all__ = ['MethodsResponse', 'StrategyEntry']
