"""Wire models for the curation ingest pipeline.

Kept in their own module so :mod:`src.services.curation.ingest` and
:mod:`src.services.curation.ingest_batch` can both depend on them
without importing each other at module scope.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# Stable machine error codes, served alongside the free-text
# ``error`` message on a failed ingest item. Not an exhaustive enum on
# the wire model (a future failure mode should still surface a message
# even if this list hasn't been extended for it) but every current
# writer uses one of these.
ERROR_KIND_EMPTY = 'empty'
ERROR_KIND_UNSERVABLE_PATH = 'unservable_path'
ERROR_KIND_UNSUPPORTED_TYPE = 'unsupported_type'
ERROR_KIND_TOO_LARGE = 'too_large'
ERROR_KIND_DECODE_FAILED = 'decode_failed'
ERROR_KIND_DETECTOR_INFER = 'detector_infer'
ERROR_KIND_BULK_INDEX = 'bulk_index'


class IngestSummary(BaseModel):
    successful: int = 0
    duplicates: int = 0
    failed: int = 0
    labels_imported: int = 0
    mismatches: int = 0
    missed_labels: int = 0
    unmatched_detections: int = 0
    crops_indexed: int = 0
    # F-43: a configured secondary detector (OP_INGEST_SECONDARY_DETECTOR_MODEL)
    # that errors (e.g. DEADLINE_EXCEEDED) per-item used to be swallowed --
    # logged at 'warning' only, with the image otherwise ingesting
    # 'successful' via the primary detector alone. A misconfigured or
    # unreachable secondary model was therefore invisible in the response.
    # Count of images where the secondary call failed (ingest still
    # succeeds on the primary detector's output alone).
    secondary_detector_failures: int = 0


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
    # Items this ingest seeded ``pending_detection`` for the region worker
    # (0 when no region profile is active).
    n_region_queued: int = 0
    error: str | None = None
    error_kind: str | None = None
    # The client-supplied identifier, for a byte-upload ingest
    # where image_path is now the server-persisted path.
    source_identifier: str | None = None
    # F-43: set when a configured secondary detector call failed for this
    # image (ingest still succeeds using the primary detector alone). Was
    # previously logged at 'warning' only and dropped -- invisible on the
    # wire even though the operator asked for a secondary classifier and
    # it never ran.
    secondary_detector_error: str | None = None


class BatchIngestResult(BaseModel):
    status: Literal['success', 'partial', 'error'] = 'success'
    summary: IngestSummary = Field(default_factory=IngestSummary)
    results: list[IngestResult] = Field(default_factory=list)
    # Model-vs-label disagreement records (``detect_mismatches``); see the
    # DISAGREEMENT_* kinds in src.services.curation.label_import.
    disagreements: list[dict[str, Any]] = Field(default_factory=list)


__all__ = ['BatchIngestResult', 'IngestResult', 'IngestSummary']
