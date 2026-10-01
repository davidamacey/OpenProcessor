"""The one place a :class:`VlmLabeler` is constructed (W9.0/M4).

``labeler_for(endpoint, pack)`` builds -- and caches -- a labeler from a
resolved :class:`~src.services.labeling.vlm_endpoints.VlmEndpoint` and a
prompt pack. Every API route and the detection worker call it, so the
endpoint's image cap, JSON mode, timeout, rate limit and identity are applied
identically everywhere (``tests/test_vlm_construction_sites.py`` pins that
nothing else constructs one).

The cache key covers everything a labeler is built from: the endpoint's
``ref`` (revision-pinned, or body-hashed for ``env``/drafts), the resolved
JSON mode, the probe marker (a re-probe -- which is also how a rotated
secret is picked up -- changes it), and the pack's name and content hash.
Every labeler also re-checks its endpoint before sending, at most once per
:data:`_RECHECK_S` (:func:`_assert_may_connect`): a host's DNS can change
after the endpoint was validated, and a worker's or a cached labeler is long
lived. A refusal is sticky until the host is acceptable again (fail closed).
A labeler that a newer key replaced is closed only after ``timeout_s + 10``
seconds, so an in-flight call is never cut off mid-request.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from src.core.logging import get_logger
from src.services.labeling.vlm_client import VlmIdentity
from src.services.labeling.vlm_endpoints import VlmEndpointUnavailableError, resolve_api_key
from src.services.labeling.vlm_labeler import VlmLabeler
from src.services.labeling.vlm_url_policy import compute_locality, url_denial


if TYPE_CHECKING:
    from collections.abc import Callable

    from src.services.labeling.vlm_endpoints import VlmEndpoint
    from src.services.labeling.vlm_prompts import PromptPack

logger = get_logger(__name__)

_CACHE_MAX = 32
_CLOSE_GRACE_S = 10.0
_RECHECK_S = 30.0
_LABELERS: OrderedDict[tuple[Any, ...], VlmLabeler] = OrderedDict()
_CLOSING: set[asyncio.Task[None]] = set()


class VlmEndpointDeniedError(VlmEndpointUnavailableError):
    """The endpoint's URL is on the never-allowed list (SSRF policy)."""


def _pack_hash(pack: PromptPack) -> str:
    return hashlib.sha256(json.dumps(pack.to_dict(), sort_keys=True).encode()).hexdigest()[:12]


def _cache_key(endpoint: VlmEndpoint, pack: PromptPack) -> tuple[Any, ...]:
    return (
        endpoint.ref,
        endpoint.json_mode_on,
        endpoint.probe_marker,
        pack.name,
        _pack_hash(pack),
        # The labeler carries the pack object, and the provenance stamp
        # reads the revision tag off it: two revisions with identical
        # bodies must not share a labeler or one would stamp the other's.
        getattr(pack, '_resolved_revision', None),
    )


def _schedule_close(labeler: VlmLabeler) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return  # no loop (sync caller): the client is released with the labeler

    async def _close_later() -> None:
        await asyncio.sleep(labeler.timeout_s + _CLOSE_GRACE_S)
        try:
            await labeler.aclose()
        except Exception as exc:
            logger.warning('vlm_labeler_close_failed', error=str(exc))

    task = loop.create_task(_close_later())
    _CLOSING.add(task)
    task.add_done_callback(_CLOSING.discard)


def _assert_may_connect(endpoint: VlmEndpoint) -> None:
    """Raise :class:`VlmEndpointDeniedError` unless the endpoint's host is,
    NOW, one this deployment may send crops to: not a never-allowed address,
    and (when it resolves outside the deployment) permitted by
    ``OP_VLM_EXTERNAL_POLICY`` and acknowledged on the endpoint. The same
    :func:`check_external` the validator runs; the ``env`` built-in is the
    operator's own choice and is not re-checked."""
    from src.services.config_store.vlm_validation import check_external

    if endpoint.source == 'env':
        return
    body = endpoint.body
    denial = url_denial(body.base_url)
    if denial is not None:
        raise VlmEndpointDeniedError(denial.reason)
    refusals = check_external(body, compute_locality(body.base_url), is_env=False)
    if refusals:
        raise VlmEndpointDeniedError(refusals[0].message)


def _egress_check(endpoint: VlmEndpoint) -> Callable[[], None]:
    """The labeler's pre-send check: :func:`_assert_may_connect`, throttled
    to once per :data:`_RECHECK_S` while it keeps passing."""
    checked_at = time.monotonic()

    def check() -> None:
        nonlocal checked_at
        if time.monotonic() - checked_at < _RECHECK_S:
            return
        _assert_may_connect(endpoint)
        checked_at = time.monotonic()

    return check


def _build(endpoint: VlmEndpoint, pack: PromptPack) -> VlmLabeler:
    body = endpoint.body
    is_env = endpoint.source == 'env'
    _assert_may_connect(endpoint)
    key = resolve_api_key(body.api_key_ref, is_env_builtin=is_env)
    if key is None and body.api_key_ref is not None and not is_env:
        msg = f'endpoint {endpoint.name!r}: the secret {body.api_key_ref!r} is missing or empty'
        raise VlmEndpointUnavailableError(msg)
    return VlmLabeler(
        base_url=body.base_url,
        model=body.model,
        # ``EMPTY`` is the conventional placeholder for a server with no
        # auth; it is what the env built-in has always sent.
        api_key=key or 'EMPTY',
        max_images_per_call=body.max_images_per_call,
        requests_per_second=body.requests_per_second,
        timeout_s=body.timeout_s,
        pack=pack,
        hard_cap=body.max_images_per_call,
        json_mode=endpoint.json_mode_on,
        open_images_per_call=body.effective_open_images,
        identity=VlmIdentity(endpoint_ref=endpoint.ref, model=endpoint.model_id),
        egress_check=_egress_check(endpoint),
    )


def labeler_for(endpoint: VlmEndpoint, pack: PromptPack) -> VlmLabeler:
    """The cached labeler for ``(endpoint, pack)``; built on first use."""
    key = _cache_key(endpoint, pack)
    cached = _LABELERS.get(key)
    if cached is not None:
        _LABELERS.move_to_end(key)
        return cached
    labeler = _build(endpoint, pack)
    _LABELERS[key] = labeler
    while len(_LABELERS) > _CACHE_MAX:
        _schedule_close(_LABELERS.popitem(last=False)[1])
    return labeler


def build_uncached_labeler(endpoint: VlmEndpoint, pack: PromptPack) -> VlmLabeler:
    """A labeler the caller owns and must ``aclose()`` (the worker's
    per-runtime labeler, whose lifetime is its runtime's)."""
    return _build(endpoint, pack)


def reset_labeler_cache() -> None:
    """Test-only."""
    _LABELERS.clear()


__all__ = [
    'VlmEndpointDeniedError',
    'VlmIdentity',
    'build_uncached_labeler',
    'labeler_for',
    'reset_labeler_cache',
]
