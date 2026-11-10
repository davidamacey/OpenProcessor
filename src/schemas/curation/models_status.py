"""Wire shape of ``GET {P}/models/status``: one card per model or external service."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelClassMappingCounts(BaseModel):
    mapped_count: int
    unmapped: list[str]


class ModelStatusEntry(BaseModel):
    """A Triton model, the segmenter or a VLM endpoint.

    The fields after ``unloadable`` belong to one kind only; the route sets
    ``response_model_exclude_unset`` so a row never gains a key it did not have.
    ``extra='allow'`` keeps any key a producer adds that is not listed here.
    """

    model_config = ConfigDict(extra='allow')

    name: str
    friendly_name: str
    role: str
    kind: str
    model_type: str
    status: str
    version: str | None = None
    inference_count: int | None = None
    exec_count: int | None = None
    inference_failed: int | None = None
    avg_latency_ms: float | None = None
    last_error: str | None = None
    endpoint: str | None = None
    unloadable: bool | None = None
    optional: bool | None = None
    # triton rows
    is_region_protected: bool | None = None
    requires_force_to_unload: bool | None = None
    job_id: str | None = None
    promoted_at: str | None = None
    project: str | None = None
    shared: bool | None = None
    class_mapping: ModelClassMappingCounts | None = None
    owned: bool | None = None
    sharing_revision: int | None = None
    # vlm rows
    model: str | None = None
    active: bool | None = None
    active_in: list[str] | None = None


class ModelsStatusResponse(BaseModel):
    models: list[ModelStatusEntry]
