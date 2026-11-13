"""Request model of ``POST /label_ops/accept_machine``. Leaf module."""

from __future__ import annotations

from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

from src.services.curation.item_filter import ItemFilter


Source = Literal['detector', 'vlm']


class AcceptMachineRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    dry_run: bool = True  # writing needs an explicit false
    min_confidence: float = Field(
        ge=0.5,
        le=1.0,
        description="The bar. The detector's own score, or the VLM's confidence category "
        '(high 0.92, medium 0.70, low 0.40).',
    )
    sources: list[Source] = Field(
        default_factory=lambda: list(get_args(Source)),
        min_length=1,
        description='Whose machine labels to accept.',
    )
    filter: ItemFilter | None = Field(
        default=None, description='Only items this filter matches; none for every item.'
    )
