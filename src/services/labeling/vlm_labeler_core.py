"""Transport core of the VLM labeler: construction, lifecycle, the chat POST and the health probe.

Split out of ``vlm_labeler.py``. The operation families (classify, verify, visibility,
combined) subclass :class:`VlmLabelerCore`; ``vlm_labeler.VlmLabeler`` composes them.
"""

from __future__ import annotations

import base64
import time
from typing import TYPE_CHECKING, Any, Self

from src.config import RegionFields, get_region_fields
from src.core.logging import get_logger
from src.services.curation.ops_metrics import record_vlm_request
from src.services.labeling.vlm_client import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_MAX_IMAGES_PER_CALL,
    DEFAULT_MODEL,
    DEFAULT_OPEN_IMAGES_PER_CALL,
    DEFAULT_REQUESTS_PER_SECOND,
    RETRY_MAX_ATTEMPTS,
    VlmIdentity,
    _TokenBucket,
    build_auth_headers,
    build_http_client,
    extract_message_content,
    post_chat_with_retry,
    record_chat_exchange,
)
from src.services.labeling.vlm_models import VlmHealth
from src.services.labeling.vlm_prompts import GENERIC_ITEM_PACK, PromptPack
from src.utils.upstream_errors import describe_upstream_error


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    import httpx


logger = get_logger(__name__)


# vLLM's ``json_object`` grammar only admits an object, so every batched
# class prompt asks for the per-image array wrapped in ``{"results": ...}``.
_RESULTS_ENVELOPE = 'Wrap the array in one JSON object: {"results": [ ...one entry per image... ]}.'


def _b64_jpeg(data: bytes) -> str:
    """Base64-encode JPEG bytes (no data: prefix)."""

    return base64.b64encode(data).decode('ascii')


class VlmLabelerCore:
    """Async client for an OpenAI-compatible vision-chat VLM endpoint."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        api_key: str = DEFAULT_API_KEY,
        max_images_per_call: int = DEFAULT_MAX_IMAGES_PER_CALL,
        requests_per_second: float = DEFAULT_REQUESTS_PER_SECOND,
        # 60s was tuned for an under-loaded upstream. Under heavy
        # concurrency, per-call latency can climb to 30-90s when KV
        # cache is saturated. 240s gives the worst-case batched
        # verify-and-read call room to finish without dropping the
        # request.
        timeout_s: float = 240.0,
        client: httpx.AsyncClient | None = None,
        pack: PromptPack = GENERIC_ITEM_PACK,
        fields: RegionFields | None = None,
        *,
        hard_cap: int | None = None,
        json_mode: bool = True,
        open_images_per_call: int | None = None,
        identity: VlmIdentity | None = None,
        egress_check: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        if base_url and not model:
            raise ValueError(
                f'OP_VLM_MODEL is required when a VLM URL is set (base_url={base_url!r}); '
                'there is no default model id.'
            )
        if max_images_per_call < 1:
            raise ValueError('max_images_per_call must be >= 1')
        # Hard cap = the endpoint's own per-prompt image limit (W9:
        # ``labeler_for`` passes the registered endpoint's
        # ``max_images_per_call``); ``None`` keeps the deployment env cap
        # (OP_VLM_MAX_IMAGES_PER_CALL), so even an explicit caller value
        # can't exceed what the serving engine accepts per prompt.
        _max_hard = DEFAULT_MAX_IMAGES_PER_CALL if hard_cap is None else hard_cap
        if max_images_per_call > _max_hard:
            logger.warning(
                'vlm_labeler.max_images_clamped',
                requested=max_images_per_call,
                clamped_to=_max_hard,
                reason=(
                    f'vlm_labeler hard cap is {_max_hard} (the endpoint image limit); '
                    "align it with your VLM deployment's per-prompt image limit."
                ),
            )
            max_images_per_call = _max_hard

        self._egress_check = egress_check
        self.json_mode = json_mode
        self.open_images_per_call = (
            DEFAULT_OPEN_IMAGES_PER_CALL if open_images_per_call is None else open_images_per_call
        )
        self.identity = identity or VlmIdentity(endpoint_ref='', model=model)
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_key = api_key
        self.max_images_per_call = max_images_per_call
        self.requests_per_second = requests_per_second
        self.timeout_s = timeout_s
        self._pack = pack
        self._fields = fields or get_region_fields()
        # Optional class-name list used by ``label_combined`` callers so
        # they don't have to thread the registry through every call
        # site. Set externally after init. ``name_to_id`` maps the
        # resolved name back to the registry's authoritative class_id —
        # needed because reply.class_id is the *index* into
        # ``class_names`` (which is filtered for non-deprecated
        # entries), not a registry id.
        self.class_names: list[str] = []
        self.name_to_id: dict[str, int] = {}

        self._bucket = _TokenBucket(rate=requests_per_second)
        self._client = client or build_http_client(timeout_s)
        self._owns_client = client is None

    # ----- lifecycle -----

    def client_closer(self) -> Callable[[], Awaitable[None]]:
        """A callable that closes the owned HTTP client and does not reference
        this labeler (so it can run from the labeler's own finalizer)."""
        client = self._client
        if not self._owns_client:

            async def _nothing() -> None:
                return None

            return _nothing
        return client.aclose

    async def aclose(self) -> None:
        """Close the underlying httpx client (if owned)."""

        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    # ----- HTTP plumbing -----

    def _json_mode_kwargs(self) -> dict[str, Any]:
        """``response_format: json_object`` for every payload, unless the
        endpoint rejects it (W9: some OpenAI-compatible servers 400 on it).
        The one place that key is added."""
        return {'response_format': {'type': 'json_object'}} if self.json_mode else {}

    @property
    def _headers(self) -> dict[str, str]:
        return build_auth_headers(self.api_key)

    async def _post_chat(
        self, payload: dict[str, Any], *, attempts: int = RETRY_MAX_ATTEMPTS
    ) -> dict[str, Any]:
        """POST /chat/completions with retry on 5xx + connection errors.

        ``egress_check`` (the factory's) runs first and raises to refuse the
        send: a host's DNS can change after the endpoint was validated."""

        started = time.monotonic()
        try:
            if self._egress_check is not None:
                await self._egress_check()
            url = f'{self.base_url}/chat/completions'
            response = await post_chat_with_retry(
                self._client, url, self._headers, payload, self._bucket, attempts=attempts
            )
        except Exception as exc:
            record_vlm_request(self.model, 'error', time.monotonic() - started)
            record_chat_exchange(payload, None, describe_upstream_error(exc))
            raise
        record_vlm_request(self.model, 'ok', time.monotonic() - started, response)
        record_chat_exchange(payload, response)
        return response

    # ----- public API -----

    async def health(self) -> VlmHealth:
        """Return a lightweight reachability probe.

        Issues a 1-token "reply OK" prompt; if any error happens the probe
        marks the service unreachable and surfaces ``last_error``.
        """

        payload = {
            'model': self.model,
            'messages': [{'role': 'user', 'content': 'reply with the single word OK'}],
            'max_tokens': 4,
            'temperature': 0.0,
        }
        try:
            resp = await self._post_chat(payload, attempts=1)
            _ = extract_message_content(resp)
            return VlmHealth(reachable=True, model=self.model, last_error=None)
        except Exception as exc:
            logger.warning(
                'vlm_labeler.health_failed',
                model=self.model,
                base_url=self.base_url,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return VlmHealth(
                reachable=False, model=self.model, last_error=describe_upstream_error(exc)
            )
