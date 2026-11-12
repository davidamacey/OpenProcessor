"""The Valkey set of images whose stored counts must be recomputed (R-13).

A writer that changed an item adds its image id (``SADD``); a periodic drain claims the whole set
with one ``RENAME`` (atomic, so a mark made afterwards lands in a fresh set and is never lost),
recomputes the counts of those images from the items and removes them from the claimed set as it
goes. Adding the same image again is free, which is the coalescing: a label operation over 20,000
items costs one set member per image, not a task per write.

The set is keyed by the bound project's slug, so projects never share one.

* SADD: https://valkey.io/commands/sadd/
* RENAME: https://valkey.io/commands/rename/
* SRANDMEMBER: https://valkey.io/commands/srandmember/
* SREM: https://valkey.io/commands/srem/
* SMEMBERS: https://valkey.io/commands/smembers/
* EXPIRE: https://valkey.io/commands/expire/
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import redis
import redis.asyncio as aioredis

from src.config.env import EnvError
from src.config.project_context import current_project
from src.config.queue import QueueSettings
from src.core.logging import get_logger
from src.tasks.runtime.event_wakeup import client_kwargs


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

logger = get_logger(__name__)

KEY_PREFIX: Final = 'op:image_counts:'
_TIMEOUT_S: Final = 1.0
#: How long the sweep leaves an image alone after the images index refused its count write.
PARKED_TTL_S: Final = 3600

#: The item fields the stored counts depend on: a write that touches none of them cannot change a
#: count (``counts.items``, ``labeled``, ``validated``, ``unlabeled``, ``in_holdout``).
COUNT_FIELDS: Final = frozenset({'class_id', 'class_validated', 'class_excluded', 'test_holdout'})


def dirty_key() -> str:
    return f'{KEY_PREFIX}dirty:{current_project().record.slug}'


def claimed_key() -> str:
    return f'{KEY_PREFIX}claimed:{current_project().record.slug}'


def refused_key() -> str:
    return f'{KEY_PREFIX}refused:{current_project().record.slug}'


def valkey_client() -> aioredis.Redis:
    """A Valkey client for one operation (the caller closes it)."""
    settings = QueueSettings.from_env()
    return aioredis.Redis.from_url(
        settings.url,
        socket_timeout=_TIMEOUT_S,
        socket_connect_timeout=_TIMEOUT_S,
        **client_kwargs(settings),
    )


class CountsMarkError(RuntimeError):
    """Neither Valkey nor the images index took the mark of a changed image."""


async def mark_dirty(opensearch: Any, image_ids: Iterable[str]) -> int:
    """Add ``image_ids`` to the bound project's dirty set; returns how many were marked.

    Called after the items write it describes has committed. If Valkey does not answer, the
    images are marked through the images index instead: their ``counts_at`` is cleared, which is
    exactly what the periodic sweep looks for, so the next drain recounts them, and until then a
    list treats the scope as uncounted and reads live counts (``image_view.counts_ready``). Only
    when that write fails too is the mark lost, and then the error is raised (a
    :class:`CountsMarkError` or the index's own error), never swallowed.
    """
    ids = sorted({i for i in image_ids if i})
    if not ids:
        return 0
    key = dirty_key()
    client = None
    try:
        client = valkey_client()
        await client.sadd(key, *ids)
    except (redis.RedisError, EnvError, OSError):
        logger.error('image_counts_mark_failed', images=len(ids), exc_info=True)
        await _clear_counts_at(opensearch, ids)
    finally:
        if client is not None:
            await client.aclose()
    return len(ids)


async def _clear_counts_at(opensearch: Any, image_ids: list[str]) -> None:
    from src.config.curation_accessors import images_index

    index = images_index()
    body: list[dict[str, Any]] = []
    for image_id in image_ids:
        body.append({'update': {'_index': index, '_id': image_id, 'retry_on_conflict': 3}})
        body.append({'doc': {'counts_at': None}})
    resp = await opensearch.bulk(body=body, refresh=False)
    refused = [
        i
        for i, item in zip(image_ids, resp.get('items') or [], strict=True)
        if int((item.get('update') or {}).get('status', 0)) not in (200, 201, 404)
    ]
    if refused:
        msg = f'{len(refused)} of {len(image_ids)} images could not be marked for a recount'
        raise CountsMarkError(msg)


def changed_image_ids(written: Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> list[str]:
    """The image ids of the ``(stored source, update)`` pairs whose update touches a count field."""
    ids: list[str] = []
    for source, update in written:
        if COUNT_FIELDS.isdisjoint(update):
            continue
        image_id = source.get('image_id')
        if image_id:
            ids.append(str(image_id))
        else:
            logger.warning('image_counts_source_without_image_id')
    return ids


async def mark_items_created(opensearch: Any, index: str, image_ids: Iterable[str]) -> None:
    """Mark the images that just got items in ``index``; a write to any index but the bound
    project's items index changes no image count."""
    from src.config.curation_accessors import items_index

    if index == items_index():
        await mark_dirty(opensearch, image_ids)


async def mark_items_written(
    opensearch: Any, index: str, written: Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]]
) -> None:
    """Mark the images of the updates an OCC helper just committed on ``index``, for the updates
    that touch a count field."""
    await mark_items_created(opensearch, index, changed_image_ids(written))


class DirtyImages:
    """The drain's side of the set: claim, read a batch, remove a batch."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def claim(self) -> bool:
        """Move the dirty set to the claimed set; ``True`` when there is a claimed set to work.

        A claimed set left by a drain that died is worked first (the set is never overwritten),
        and the dirty set stays for the next claim.
        """
        if await self._client.exists(claimed_key()):
            return True
        try:
            await self._client.rename(dirty_key(), claimed_key())
        except redis.ResponseError as exc:
            if 'no such key' in str(exc).lower():
                return False
            raise
        return True

    async def batch(self, size: int) -> list[str]:
        members = await self._client.srandmember(claimed_key(), size)
        return [str(m) for m in members or []]

    async def done(self, image_ids: list[str]) -> None:
        if image_ids:
            await self._client.srem(claimed_key(), *image_ids)

    async def requeue(self, image_ids: list[str]) -> None:
        """Move ``image_ids`` from the claimed set back to the dirty set (a retry on a later claim).

        Added to the dirty set first, so a failure between the two calls duplicates a mark and
        never loses one.
        """
        if image_ids:
            await self._client.sadd(dirty_key(), *image_ids)
            await self._client.srem(claimed_key(), *image_ids)

    async def park(self, image_ids: list[str]) -> None:
        """Keep the sweep off ``image_ids`` for :data:`PARKED_TTL_S` (the whole set expires that
        long after the last addition, so a parked image is tried again about an hour later)."""
        if image_ids:
            await self._client.sadd(refused_key(), *image_ids)
            await self._client.expire(refused_key(), PARKED_TTL_S)

    async def parked(self) -> list[str]:
        return sorted(str(m) for m in await self._client.smembers(refused_key()))
