"""Embed one item inside the request that wrote it (drawn box, box edit).

A job would be the wrong tool: ``reprocess`` is an exclusive job kind, so a second box drawn while
one runs would be refused ``job_busy``. The write has already stood by the time this runs, so a
failure leaves the item ``deferred`` with a served message and never undoes the write; an
``embed`` reprocess finishes it later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from opensearchpy.exceptions import OpenSearchException
from tritonclient.utils import InferenceServerException

from src.config import get_curation_config, get_region_fields
from src.core.logging import get_logger
from src.domain.embedding_state import DEFERRED, EmbeddingState
from src.services.curation.reprocess_embed import EmbedTarget, reembed_items
from src.services.curation.reprocess_targets import items_by_terms
from src.services.embedding_spaces.readers import item_write_fields


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

logger = get_logger(__name__)

#: What the encoder, its transport and the index raise; anything else is a bug and surfaces.
_EMBED_ERRORS = (OpenSearchException, InferenceServerException, OSError, RuntimeError, ValueError)

#: Frame and region vectors do not depend on the item's own box.
CROP_PART = frozenset({'crop'})

NO_ENCODER_MESSAGE = (
    'Saved. The image encoder is not available, so the item is not embedded yet; '
    'run "Embed missing items" to finish it.'
)
EMBED_FAILED_MESSAGE = (
    'Saved. The item could not be embedded just now; run "Embed missing items" to finish it.'
)


@dataclass(frozen=True)
class EmbedOutcome:
    state: EmbeddingState
    message: str | None


async def embed_item_inline(
    opensearch: AsyncOpenSearch, pe: Any | None, crop_id: str
) -> EmbedOutcome:
    """Embed ``crop_id`` now. ``embedded`` on success; ``deferred`` with the reason otherwise."""
    if pe is None:
        return EmbedOutcome(DEFERRED, NO_ENCODER_MESSAGE)
    cfg = get_curation_config()
    F = get_region_fields()
    try:
        rows = await items_by_terms(
            opensearch,
            'crop_id',
            [crop_id],
            index=cfg.items_index,
            includes=['image_id', 'image_path', 'bbox_norm', 'cluster_partition_id', F.boxes],
        )
        targets = [
            EmbedTarget(
                image_id=src.get('image_id') or '',
                image_path=src.get('image_path') or '',
                items=[(cid, src)],
            )
            for cid, src in rows
        ]
        counts = await reembed_items(opensearch, pe, targets, parts=CROP_PART, only_missing=True)
    except _EMBED_ERRORS as exc:  # the write stands whatever the encoder or the read does
        logger.warning('inline_embed_failed', crop_id=crop_id, error=str(exc))
        return EmbedOutcome(DEFERRED, EMBED_FAILED_MESSAGE)
    if not counts['crop_written']:
        return EmbedOutcome(DEFERRED, EMBED_FAILED_MESSAGE)
    state_field = (await item_write_fields()).state_field
    stored = await opensearch.get(index=cfg.items_index, id=crop_id, _source_includes=[state_field])
    return EmbedOutcome(stored['_source'].get(state_field, DEFERRED), None)
