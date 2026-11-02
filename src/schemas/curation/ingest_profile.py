"""Body of a stored ``ingest_profile``: the detector (and optional secondary
classifier) a project's ingest runs.

Only the ingest-relevant knobs are exposed. A field left ``None`` inherits the
deployment's ``OP_INGEST_PRIMARY_*`` / ``OP_INGEST_SECONDARY_*`` value, and the
profile's ``name`` (hence the ``class_source`` vocabulary) is never settable, so
it cannot fork per project. A stored profile never assigns registry class ids:
its detections are proposals, or are resolved by name through the ingest policy.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.utils.class_names import confine_labels_path


class IngestDetectorSpec(BaseModel):
    model_config = ConfigDict(extra='forbid')

    model: str = Field(min_length=1)
    version: str = '1'
    input_size: int | None = Field(default=None, ge=32, le=4096)
    # labels.txt-style file naming the model's classes; empty reads the one in
    # the model's own Triton directory.
    labels_path: str = ''
    confidence_floor: float | None = Field(default=None, ge=0.0, le=1.0)
    batch_limit: int | None = Field(default=None, ge=1, le=256)

    @field_validator('labels_path')
    @classmethod
    def _labels_path_in_model_repository(cls, value: str) -> str:
        if value:
            confine_labels_path(value)
        return value


class IngestProfileBody(BaseModel):
    model_config = ConfigDict(extra='forbid')

    primary: IngestDetectorSpec
    secondary: IngestDetectorSpec | None = None


__all__ = ['IngestDetectorSpec', 'IngestProfileBody']
