"""Wire models for ``/ingest/objects`` (ingest of objects held in an S3-compatible store)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ObjectItem(BaseModel):
    """One object version. ``etag`` and ``version_id`` pin the version when known;
    ``size`` (bytes) saves a HEAD request."""

    model_config = ConfigDict(extra='forbid')

    bucket: str = Field(min_length=1, max_length=255)
    key: str = Field(min_length=1, max_length=1024)
    etag: str = Field(default='', max_length=256)
    version_id: str | None = Field(default=None, max_length=1024)
    size: int | None = Field(default=None, ge=0)


class IngestObjectsRequest(BaseModel):
    """The server fetches each object (credentials come from the deployment, never
    from the request) and ingests it as an upload would be: staged under the
    project's upload root, the object reference stored as ``source_identifier``."""

    model_config = ConfigDict(extra='forbid')

    items: list[ObjectItem] = Field(min_length=1)
    source: str = Field(default='s3', min_length=1, max_length=120)
    ingest_run_id: str | None = Field(default=None, max_length=200)


class ObjectPresignRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    source_identifier: str = Field(min_length=6, max_length=2048)
    expires_in: int = Field(default=900, ge=60, le=43200)


class ObjectPresignResponse(BaseModel):
    url: str
    expires_in: int
