"""Wire-shaped models for a VLM endpoint's stored body and probe record
(any_domain_plan.md §7.8.1). Pure pydantic: no I/O, no router imports, so
the service layer, the worker and the routers all share one definition.

Field *ranges* are deliberately NOT pydantic constraints: an out-of-range
value must come back as a ``vlm_field_range`` :class:`ValidationIssue`
(never a bare FastAPI 422), so :func:`~src.services.config_store.vlm_validation.validate_vlm_endpoint`
owns them (:data:`FIELD_RANGES`).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


JsonMode = Literal['auto', 'on', 'off']

#: ``field -> (min, max)``, inclusive; served in the schema rows too.
FIELD_RANGES: dict[str, tuple[float, float]] = {
    'max_images_per_call': (1, 64),
    'open_images_per_call': (1, 64),
    'timeout_s': (5, 900),
    'requests_per_second': (0.1, 10000),
}


class VlmEndpointBody(BaseModel):
    """What is stored for an endpoint. The API key itself is never part of
    it: ``api_key_ref`` only names where to find it (W9.9)."""

    model_config = ConfigDict(extra='forbid')

    base_url: str
    model: str
    api_key_ref: str | None = None
    max_images_per_call: int = 8
    open_images_per_call: int | None = 3
    timeout_s: float = 240.0
    requests_per_second: float = 500.0
    json_mode: JsonMode = 'auto'
    allow_external: bool = False
    catalog_id: str | None = None

    def normalized(self) -> VlmEndpointBody:
        """``base_url`` without a trailing ``/`` (the stored form)."""
        return self.model_copy(update={'base_url': self.base_url.strip().rstrip('/')})

    @property
    def effective_open_images(self) -> int:
        """``open_images_per_call`` resolved (``None`` = ``min(3, cap)``),
        clamped to the endpoint cap."""
        wanted = 3 if self.open_images_per_call is None else self.open_images_per_call
        return max(1, min(wanted, self.max_images_per_call))


class VlmProbeRecord(BaseModel):
    """The last probe of one endpoint (``vlm_probe:<name>``). ``issues`` are
    serialised ``ValidationIssue`` dicts (the router types them)."""

    model_config = ConfigDict(extra='ignore')

    ok: bool
    probed_at: str
    latency_ms: float | None = None
    models_listed: list[str] = Field(default_factory=list)
    model_listed: bool | None = None
    root: str | None = None
    max_model_len: int | None = None
    vision_ok: bool | None = None
    json_mode_supported: bool | None = None
    reasoning_channel: bool | None = None
    image_tokens: int | None = None
    max_images_ok: bool | None = None
    issues: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def error_codes(self) -> list[str]:
        return [i['code'] for i in self.issues if i.get('severity') == 'error']


__all__ = ['FIELD_RANGES', 'JsonMode', 'VlmEndpointBody', 'VlmProbeRecord']
