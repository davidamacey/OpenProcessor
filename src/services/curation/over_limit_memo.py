"""Items a drain skips without spending model work: they hold too many nested documents.

An item at or over the nested-document limit (``src/domain/nested_limit.py``, OpenSearch k-NN
issue 3623) refuses every machine write, so a drain that picks it up would run its models, fail
the write and pick it up again on the next cycle, for ever. A drain (the region worker, the VLM
worker) reads the box lists and assist attributes of the page it fetched, drops such items
before any model call, remembers their ids for a while (``excluded`` goes into the next
query, so they neither hold a cycle busy nor fill a page) and counts them. The memo expires,
so an item repaired in the meantime is looked at again.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from src.config.nested_limit import NESTED_DOCS_LIMIT
from src.domain.nested_limit import box_vector_paths, nested_doc_count
from src.services.embedding_spaces.readers import vector_source_excludes


if TYPE_CHECKING:
    from collections.abc import Callable, Collection

DEFAULT_TTL_S = 600.0
MAX_REMEMBERED = 100_000


class OverLimitMemo:
    def __init__(
        self, *, ttl_s: float = DEFAULT_TTL_S, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._ttl_s = ttl_s
        self._clock = clock
        self._until: dict[str, float] = {}
        self.skipped = 0
        """Items skipped over the nested limit since the process started (not distinct)."""

    def excluded(self) -> list[str]:
        """The ids to keep out of the next pending query."""
        now = self._clock()
        for crop_id in [c for c, until in self._until.items() if until <= now]:
            del self._until[crop_id]
        return list(self._until)

    def split(
        self, hits: list[dict[str, Any]], paths: Collection[str]
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """``(hits to work, ids skipped)``: a hit whose ``_source`` holds ``NESTED_DOCS_LIMIT``
        nested documents or more (vectors not fetched: ``paths`` are the box-vector lists each
        counted one entry per box) is skipped and remembered."""
        kept: list[dict[str, Any]] = []
        over: list[str] = []
        for hit in hits:
            src = hit.get('_source') or {}
            if nested_doc_count(src, unread_box_vector_paths=paths) >= NESTED_DOCS_LIMIT:
                over.append(str(hit['_id']))
            else:
                kept.append(hit)
        if over and len(self._until) < MAX_REMEMBERED:
            until = self._clock() + self._ttl_s
            self._until.update(dict.fromkeys(over, until))
        self.skipped += len(over)
        return kept, over


MEMO = OverLimitMemo()
"""The process's memo (one drain process per worker)."""


async def bound_project_box_vector_paths() -> frozenset[str]:
    """The box-vector lists of the bound project, for :meth:`OverLimitMemo.split`."""
    return box_vector_paths(await vector_source_excludes())


__all__ = ['MEMO', 'OverLimitMemo', 'bound_project_box_vector_paths']
