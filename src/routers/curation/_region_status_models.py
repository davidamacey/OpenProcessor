"""Wire models for ``GET /regions/statuses`` and the region cluster cards. Leaf module."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.routers.curation._region_row_models import RegionRow


class RegionStatusEntry(BaseModel):
    value: str
    label: str
    role: str
    terminal: bool
    human_writable: bool
    clears_box: bool = Field(description='A human write of this status removes the region box.')
    wants_reason: bool


class RegionBoxStateEntry(BaseModel):
    value: str
    label: str
    role: str
    tone: str = Field(description='A semantic meaning, not a CSS color.')
    human_writable: bool
    exported: bool
    dashed: bool
    dim: bool
    badge: str | None


class RegionBoxStateRoute(BaseModel):
    route: str
    states: list[str]
    new_box_default: str | None


class RegionStatusCatalog(BaseModel):
    statuses: list[RegionStatusEntry]
    confirm_status: str
    reject_status: str
    false_positive_status: str
    box_states: list[RegionBoxStateEntry]
    box_state_routes: list[RegionBoxStateRoute]


class RegionClusterCard(BaseModel):
    cluster_uid: str
    size: int = Field(description='Items with at least one box in the cluster.')
    box_count: int
    validated_count: int
    representative_crop_ids: list[str]
    representative_box_ids: list[str | None]
    representative_thumb_urls: list[str]
    representatives: list[RegionRow]


class RegionClusterList(BaseModel):
    clusters: list[RegionClusterCard]
    count: int
