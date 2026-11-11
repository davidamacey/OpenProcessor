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

from pydantic import BaseModel, ConfigDict, Field, model_validator


JsonMode = Literal['auto', 'on', 'off']
#: Who serves the endpoint. Each provider has defaults for the request parameters it accepts
#: (:data:`PROVIDER_DEFAULTS`); every one is overridable per endpoint.
#: vLLM: vLLM OpenAI-compatible server docs. Ollama: docs.ollama.com/api/openai-compatibility
#: (``response_format`` and ``temperature`` supported). Anthropic: platform.claude.com/docs/en/api/openai-sdk
#: (``json_object`` and, on current models, ``temperature`` return 400; ``anthropic-version`` header).
Provider = Literal['vllm', 'ollama', 'anthropic', 'openai_compatible']
ANTHROPIC_VERSION = '2023-06-01'

#: ``provider -> {field: default}`` applied to a body that leaves the field out.
PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    'vllm': {'json_mode': 'auto', 'temperature': 0.0},
    'ollama': {'json_mode': 'auto', 'temperature': 0.0},
    'openai_compatible': {'json_mode': 'auto', 'temperature': 0.0},
    'anthropic': {'json_mode': 'off', 'temperature': None},
}

#: ``field -> (min, max)``, inclusive; served in the schema rows too.
FIELD_RANGES: dict[str, tuple[float, float]] = {
    'max_images_per_call': (1, 64),
    'open_images_per_call': (1, 64),
    'timeout_s': (5, 900),
    'requests_per_second': (0.1, 10000),
    'temperature': (0, 2),
}


class VlmEndpointBody(BaseModel):
    """What is stored for an endpoint. The API key itself is never part of
    it: ``api_key_ref`` only names where to find it."""

    model_config = ConfigDict(extra='forbid')

    base_url: str
    model: str
    api_key_ref: str | None = None
    max_images_per_call: int = 8
    open_images_per_call: int | None = 3
    timeout_s: float = 240.0
    requests_per_second: float = 500.0
    json_mode: JsonMode = 'auto'
    #: Sent on every chat call; ``None`` leaves it out (current Claude models reject it).
    temperature: float | None = 0.0
    provider: Provider = 'vllm'
    allow_external: bool = False
    catalog_id: str | None = None

    @model_validator(mode='before')
    @classmethod
    def _provider_defaults(cls, data: Any) -> Any:
        """A body that names a provider but not a parameter gets the provider's default for it."""
        if isinstance(data, dict):
            defaults = PROVIDER_DEFAULTS.get(str(data.get('provider', 'vllm')), {})
            return {**defaults, **data}
        return data

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


__all__ = [
    'ANTHROPIC_VERSION',
    'FIELD_RANGES',
    'PROVIDER_DEFAULTS',
    'JsonMode',
    'Provider',
    'VlmEndpointBody',
    'VlmProbeRecord',
]
