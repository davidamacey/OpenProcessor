"""Wire model for ``GET /review/new_class_proposals/summary``. Leaf module."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ProposalTerm(BaseModel):
    label: str
    count: int = Field(description='What a resolve for this label matches.')
    sample_crop_ids: list[str | None]
    flag: Literal['existing_class', 'generic_parent', 'non_object'] | None
    class_id: int | None = Field(description='Set when flag is existing_class: map to it.')


class ProposalTermRulesWire(BaseModel):
    generic_terms: list[str]
    non_object_terms: list[str]
    registry_groups_are_generic: bool
    existing_classes_flagged: bool
    generic_terms_env: str
    non_object_terms_env: str


class NewClassProposalSummary(BaseModel):
    total_pending: int
    without_term: int = Field(description='Queue items with no proposed name.')
    top_terms: list[ProposalTerm] = Field(description='Terms worth creating (flag null).')
    flagged_terms: list[ProposalTerm]
    term_rules: ProposalTermRulesWire
