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
A labeler that a newer key replaced is closed only after ``timeout_s + 10``
seconds, so an in-flight call is never cut off mid-request.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from src.core.logging import get_logger
from src.services.labeling.vlm_client import VlmIdentity
from src.services.labeling.vlm_endpoints import VlmEndpointUnavailableError, resolve_api_key
from src.services.labeling.vlm_labeler import VlmLabeler
from src.services.labeling.vlm_url_policy import url_denial


if TYPE_CHECKING:
    from src.services.labeling.vlm_endpoints import VlmEndpoint
    from src.services.labeling.vlm_prompts import PromptPack

logger = get_logger(__name__)

_CACHE_MAX = 32
_CLOSE_GRACE_S = 10.0
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


def _build(endpoint: VlmEndpoint, pack: PromptPack) -> VlmLabeler:
    body = endpoint.body
    is_env = endpoint.source == 'env'
    if not is_env:
        # Re-checked on every cache miss: a stored endpoint's host can start
        # resolving to a forbidden address after it was validated.
        denial = url_denial(body.base_url)
        if denial is not None:
            raise VlmEndpointDeniedError(denial.reason)
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
