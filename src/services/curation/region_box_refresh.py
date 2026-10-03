"""Keep per-box vectors in step with a human box edit.

A box a person draws, moves, resizes or newly accepts has no valid vector (a
vector records the geometry it was computed from, so a moved box counts as
having none; see :mod:`~src.services.curation.region_box_embeddings`), and a
deleted or moved box leaves a stale entry. :func:`refresh_box_embeddings` is
the one function every human box-edit route calls after its write: it prunes
what an edit invalidated, then embeds the accepted and false-positive boxes
that have no valid vector, through the same code the ``embed`` reprocess scope
runs (:func:`~src.services.curation.reprocess_embed.reembed_items`, region
part, only missing). It runs in the edit request (GPU work in a write, never
in a read) and never fails the edit: a box it could not embed is reported as
``pending`` and the ``embed`` scope picks it up.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.config import get_curation_config
from src.config.region_fields import get_region_fields
from src.core.logging import get_logger
from src.services.curation.region_box_embeddings import missing_boxes, prune_box_embeddings
from src.services.curation.reprocess_embed import EmbedTarget, reembed_items
from src.services.curation.reprocess_targets import items_by_terms


if TYPE_CHECKING:
    from collections.abc import Iterable

    from opensearchpy import AsyncOpenSearch

logger = get_logger(__name__)

REGION_PART = frozenset({'region'})


def _encoder() -> Any | None:
    from src.main import app

    return getattr(app.state, 'pe_encoder', None)


def _pending_boxes(docs: list[tuple[str, dict[str, Any]]]) -> int:
    return sum(len(missing_boxes(src)) for _, src in docs)


async def refresh_box_embeddings(
    opensearch: AsyncOpenSearch, crop_ids: Iterable[str], *, pe: Any | None = None
) -> dict[str, int]:
    """Prune, then embed the boxes of ``crop_ids`` that lack a valid vector.

    Returns ``{'embedded': n, 'pending': m}``: boxes embedded now, and boxes
    still without a valid vector (encoder unavailable, image unreadable or the
    encoder raised).
    """
    ids = list(dict.fromkeys(crop_ids))
    if not ids:
        return {'embedded': 0, 'pending': 0}
    cfg = get_curation_config()
    await prune_box_embeddings(opensearch, index=cfg.items_index, crop_ids=ids)
    F = get_region_fields()
    includes = [
        'image_id',
        'image_path',
        F.boxes,
        f'{F.box_embeddings}.box_id',
        f'{F.box_embeddings}.bbox_norm',
    ]
    try:
        docs = await items_by_terms(
            opensearch, 'crop_id', ids, index=cfg.items_index, includes=includes
        )
    except Exception as exc:
        logger.warning('box_refresh_read_failed', error=str(exc))
        return {'embedded': 0, 'pending': 0}
    before = _pending_boxes(docs)
    encoder = pe if pe is not None else _encoder()
    if before == 0 or encoder is None:
        return {'embedded': 0, 'pending': before}
    targets = [
        EmbedTarget(
            image_id=src.get('image_id') or '',
            image_path=src.get('image_path') or '',
            items=[(cid, src)],
        )
        for cid, src in docs
        if missing_boxes(src)
    ]
    try:
        counts = await reembed_items(
            opensearch, encoder, targets, parts=REGION_PART, only_missing=True
        )
    except Exception as exc:
        logger.warning('box_refresh_embed_failed', error=str(exc))
        return {'embedded': 0, 'pending': before}
    written = counts['region_written']
    return {'embedded': written, 'pending': max(0, before - written)}


__all__ = ['refresh_box_embeddings']
