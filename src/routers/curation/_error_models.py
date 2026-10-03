"""Typed error bodies for the routes whose 4xx/5xx the contract publishes beyond
the generic config-store error. Every body is ``{"detail": {error, message,
...}}``: ``error`` a stable machine code, ``message`` for a person, then the
code-specific fields. Leaf module (raised through :func:`api_error`)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class RegionProfileUnavailableDetail(BaseModel):
    error: Literal['no_active_profile']
    message: str


class RegionProfileUnavailableResponse(BaseModel):
    """409: the route needs an active region profile and the project has none."""

    detail: RegionProfileUnavailableDetail


class DetectorUnavailableDetail(BaseModel):
    error: Literal['detector_unavailable']
    message: str


class DetectorUnavailableResponse(BaseModel):
    """503: no ingest detector (or no label list) is configured, the ingest
    configuration is invalid, or Triton cannot be asked."""

    detail: DetectorUnavailableDetail


class UnknownDetectorNamesDetail(BaseModel):
    error: Literal['unknown_detector_names']
    message: str
    unknown_names: list[str]


class UnknownDetectorNamesResponse(BaseModel):
    """422: a requested class name is not in the detector's vocabulary."""

    detail: UnknownDetectorNamesDetail


class DetectorNotServableDetail(BaseModel):
    error: Literal['detector_not_servable']
    message: str
    reasons: list[str]


class DetectorNotServableResponse(BaseModel):
    """422: the detector override cannot be served; ``reasons`` lists every problem."""

    detail: DetectorNotServableDetail


REGION_PROFILE_RESPONSES: dict[int | str, dict[str, Any]] = {
    409: {'model': RegionProfileUnavailableResponse}
}

__all__ = [
    'REGION_PROFILE_RESPONSES',
    'DetectorNotServableResponse',
    'DetectorUnavailableResponse',
    'RegionProfileUnavailableResponse',
    'UnknownDetectorNamesResponse',
]
