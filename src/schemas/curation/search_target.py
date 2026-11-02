"""Which stored vector a curation semantic search ranks against."""

from __future__ import annotations

from enum import StrEnum


class SearchTarget(StrEnum):
    """``item``: the crop's own vector (``pe_embedding``). ``box``: the per-box vectors inside
    an item (``region_box_embeddings``); the page is still one card per item."""

    ITEM = 'item'
    BOX = 'box'


__all__ = ['SearchTarget']
