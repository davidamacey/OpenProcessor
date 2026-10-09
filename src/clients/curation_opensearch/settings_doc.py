"""The shared curation-settings document: cached read and partial-merge write."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from src.clients.curation_opensearch.base import config, logger
from src.clients.optional_doc import get_doc_or_none
from src.config import CurationConfig, IndexRole, index_name


CURATION_SETTINGS_DOC_ID = 'default'
"""Fixed OpenSearch doc id the settings index always addresses -- this is a
single shared-defaults document, not a full index of many settings rows
(curation_design_rationale.md's config-dataclass philosophy: one small,
explicit piece of deployment/runtime state, not a generic key-value
store). ``'default'`` (not e.g. ``'singleton'``) because it reads naturally
alongside the field it stores (\"the defaults doc\"), and because a future
per-tenant settings doc (if this ever stops being a single shared
instance) would key by tenant id with this same literal as the
single-tenant fallback."""


# get_curation_settings is read on nearly every strategy-scoring
# request path (strategy_defaults.py, strategy_registry.py both fetch it
# per call). A 5s TTL cache avoids a GET-by-id round trip on every one of
# those, while staying short enough that a settings change is visible
# almost immediately -- and update_curation_settings below invalidates it
# immediately on write anyway, so the TTL only matters between writes.
# Keyed by index name so a caller passing a non-default cfg doesn't share
# another deployment's cached doc.
_SETTINGS_CACHE_TTL_SECONDS = 5.0
_settings_cache: dict[str, tuple[dict[str, Any], float]] = {}


def _invalidate_settings_cache(index: str) -> None:
    _settings_cache.pop(index, None)


async def get_curation_settings(client: Any, cfg: CurationConfig | None = None) -> dict[str, Any]:
    """Fetch the shared curation-settings document.

    Get-or-default-empty: a missing document (nothing has ever been PUT)
    is not an error -- it means "no shared override for any axis yet" --
    so this always returns the full envelope shape with ``defaults: {}``
    rather than raising or returning ``None``.

    An axis explicitly cleared via ``update_curation_settings(..., {axis:
    None})`` is stored as a literal ``null`` (OpenSearch's partial-doc
    merge sets a nested field to null rather than deleting the key) --
    filtered out here so a cleared axis simply doesn't appear in
    ``defaults``, identical to "never had an override."

    Cached for :data:`_SETTINGS_CACHE_TTL_SECONDS`, invalidated
    immediately by :func:`update_curation_settings` on write.
    """
    active_cfg = cfg or config
    index = index_name(active_cfg, IndexRole.SETTINGS)

    cached = _settings_cache.get(index)
    if cached is not None and time.monotonic() < cached[1]:
        return cached[0]

    source: dict[str, Any] = {}
    try:
        resp = await get_doc_or_none(client, index, CURATION_SETTINGS_DOC_ID)
        source = (resp or {}).get('_source') or {}
    except Exception as exc:
        # Mirrors image_serving.fetch_crop_source's duck-typed not-found
        # check -- avoids a hard opensearchpy import just to catch
        # NotFoundError, so a plain test mock with a raising `.get` works
        # the same way the real client does.
        msg = str(exc).lower()
        if not ('notfound' in msg or 'not found' in msg or '404' in msg):
            logger.warning('curation_settings_get_failed', error=str(exc))
    raw_defaults = source.get('defaults') or {}
    result = {
        'defaults': {k: v for k, v in raw_defaults.items() if v is not None},
        'updated_at': source.get('updated_at'),
        'updated_by': source.get('updated_by'),
    }
    _settings_cache[index] = (result, time.monotonic() + _SETTINGS_CACHE_TTL_SECONDS)
    return result


async def update_curation_settings(
    client: Any, defaults: dict[str, str | None], cfg: CurationConfig | None = None
) -> dict[str, Any]:
    """Partially merge ``defaults`` into the single shared settings doc.

    Uses OpenSearch's partial-update ``doc`` merge (recursive for object
    fields, per the update API's documented semantics) so axes not
    mentioned in this call are left untouched -- callers never need to
    read-modify-write the whole document themselves. ``doc_as_upsert``
    creates the document on the very first write. ``updated_by`` stays
    ``None`` -- there is no user-account system yet (single shared
    instance) -- but the field is written on every call so the schema
    already carries it for when one exists.

    A ``None`` value for an axis clears its shared override -- stored as
    a literal null (see :func:`get_curation_settings`'s note on why that
    read path filters it back out).

    No ``refresh=True`` -- the read-immediately-after-write below
    is a single-doc ``GET`` (not ``_search``), which OpenSearch serves
    real-time from the translog regardless of the index's refresh
    interval, so forcing a segment refresh here bought nothing but
    latency. The 5s settings cache is invalidated immediately (not left
    to expire) so this read-after-write can never return a stale value.
    """
    active_cfg = cfg or config
    index = index_name(active_cfg, IndexRole.SETTINGS)
    body = {
        'doc': {
            'defaults': defaults,
            'updated_at': datetime.now(UTC).isoformat(),
            'updated_by': None,
        },
        'doc_as_upsert': True,
    }
    await client.update(index=index, id=CURATION_SETTINGS_DOC_ID, body=body)
    _invalidate_settings_cache(index)
    return await get_curation_settings(client, cfg=active_cfg)
