"""Wire model for `GET /health`."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from src.routers.curation._region_vocabulary_models import RegionProfileSummary


# Runtime import: pydantic resolves the Literal annotation from module globals.


class HealthResponse(BaseModel):
    """Scoped ``{prefix}/health``. Project-bound keys: ``project``,
    ``opensearch.indexes``, ``registry``, ``region_profile``; the rest are
    deployment facts (also on the global ``/health``). See
    ``src/routers/curation/health.py``."""

    status: Literal['ok', 'degraded', 'down']
    project: str = Field(
        description='The bound project slug, so a late response from a previously '
        'selected project can be recognised and dropped.'
    )
    triton: dict[str, Any]
    opensearch: dict[str, Any]
    vlm: dict[str, Any]
    registry: dict[str, Any]
    region_profile: RegionProfileSummary | None = Field(
        description=(
            'The active region profile, or null when none is configured -- '
            'THE signal a client checks to decide whether region-scoped '
            'UI/routes are available.'
        )
    )
    mlflow_public_url: str | None = Field(
        default=None,
        description=(
            'The browser-reachable MLflow base URL: OP_MLFLOW_PUBLIC_URL '
            'verbatim, else the requesting host on OP_MLFLOW_PORT. '
            'A served train run may carry its own mlflow_run_url that is '
            'correct host-side but unreachable from an operator browser -- '
            'a client should build the link from this base rather than '
            'guessing a port.'
        ),
    )
