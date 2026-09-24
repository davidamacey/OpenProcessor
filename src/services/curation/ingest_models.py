"""Wire models for the curation ingest pipeline.

Kept in their own module so :mod:`src.services.curation.ingest` and
:mod:`src.services.curation.ingest_batch` can both depend on them
without importing each other at module scope.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class IngestSummary(BaseModel):
    successful: int = 0
    duplicates: int = 0
    failed: int = 0
    labels_imported: int = 0
    mismatches: int = 0
    missed_labels: int = 0
    unmatched_detections: int = 0
    crops_indexed: int = 0


class IngestResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    status: Literal['success', 'duplicate', 'failed'] = 'success'
    image_id: str = ''
    image_path: str = ''
    imohash: str = ''
    n_crops: int = 0
    crops_created: int = 0
    crops_updated: int = 0
    crops_preserved_human: int = 0
    crops_final_conflicts: int = 0
    error: str | None = None
    error_kind: str | None = None


class BatchIngestResult(BaseModel):
    status: Literal['success', 'partial', 'error'] = 'success'
    summary: IngestSummary = Field(default_factory=IngestSummary)
    results: list[IngestResult] = Field(default_factory=list)
    # Model-vs-label disagreement records (``detect_mismatches``); see the
    # DISAGREEMENT_* kinds in src.services.curation.label_import.
    disagreements: list[dict[str, Any]] = Field(default_factory=list)


__all__ = ['BatchIngestResult', 'IngestResult', 'IngestSummary']
