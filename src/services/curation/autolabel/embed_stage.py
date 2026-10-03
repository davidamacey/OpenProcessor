"""The auto-label run's optional first stage: embed the items in scope that
have no vector yet, so the clustering and VLM stages that follow work on them.

The scope is the run's ``class_id`` / ``cluster_id`` plus the shared item
filter; only items stored without a vector for a recorded reason
(``not_selected``, ``deferred``, ``failed``) are embedded, by the same code the
``embed`` reprocess scope runs. The worker has no resident encoder, so one is
built per run from ``OP_TRITON_URL`` / ``TRITON_URL`` (default
``triton-server:8001``, reachable on the compose network).
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any

from src.core.logging import get_logger
from src.services.curation.embedding_state import DEFERRED, FAILED, NOT_SELECTED
from src.services.curation.item_filter import ItemFilter
from src.services.curation.reprocess import plan_reprocess
from src.services.curation.reprocess_images import CHUNK, embed_image_chunk
from src.services.curation.reprocess_models import (
    EmbedOptions,
    ReprocessFilter,
    ReprocessRequest,
    ReprocessScopeResult,
    ReprocessTargets,
)
from src.services.curation.reprocess_targets import existing_images


if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch

logger = get_logger(__name__)

DEFAULT_TRITON_URL = 'triton-server:8001'


def triton_url() -> str:
    return os.environ.get('OP_TRITON_URL') or os.environ.get('TRITON_URL') or DEFAULT_TRITON_URL


def stage_request(
    class_id: int | None, cluster_id: int | None, item_filter: ItemFilter
) -> ReprocessRequest:
    """The ``embed`` reprocess (only the missing) for a run's scope."""
    scope = item_filter.model_copy(
        update={
            'embedding_state': [NOT_SELECTED, DEFERRED, FAILED],
            **({'class_id': class_id} if class_id is not None else {}),
            **({'cluster_id': cluster_id} if cluster_id is not None else {}),
        }
    )
    return ReprocessRequest(
        targets=ReprocessTargets(filter=ReprocessFilter(**scope.model_dump())),
        scopes=['embed'],
        embed=EmbedOptions(only_missing=True),
        dry_run=False,
    )


async def run_embed_missing_stage(
    opensearch: AsyncOpenSearch,
    *,
    class_id: int | None,
    cluster_id: int | None,
    item_filter: dict[str, Any] | None,
    progress: Any = None,
    encoder: Any = None,
) -> dict[str, Any]:
    """Embed the missing items in scope; returns the stage's summary
    (``skipped`` when there is nothing to embed, ``status: error`` when it
    failed: the run goes on and the later stages work on what has a vector).
    ``encoder`` is injectable for tests."""
    try:
        return await _embed_missing(
            opensearch, class_id, cluster_id, item_filter, progress, encoder
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning('pipeline_embed_missing_failed', error=str(exc))
        return {'status': 'error', 'error': str(exc)}


async def _embed_missing(
    opensearch: AsyncOpenSearch,
    class_id: int | None,
    cluster_id: int | None,
    item_filter: dict[str, Any] | None,
    progress: Any,
    encoder: Any,
) -> dict[str, Any]:
    request = stage_request(class_id, cluster_id, ItemFilter(**(item_filter or {})))
    plan = await plan_reprocess(opensearch, request)
    if not plan.image_ids or plan.embed is None:
        return {'skipped': True, 'reason': 'nothing to embed'}
    owned_pool: Any = None
    if encoder is None:
        from src.clients.pe_encoder import PEEncoder
        from src.clients.triton_pool import AsyncTritonPool

        owned_pool = AsyncTritonPool(url=triton_url(), pool_size=1, max_concurrent=8)
        encoder = PEEncoder(triton_pool=owned_pool)
    result = ReprocessScopeResult(scope='embed', selected=len(plan.image_ids))
    try:
        for start in range(0, len(plan.image_ids), CHUNK):
            if progress is not None:
                progress.update(processed=start, total=len(plan.image_ids))
                progress.raise_if_cancelled()
            chunk = plan.image_ids[start : start + CHUNK]
            docs = await existing_images(opensearch, chunk, index=_images_index())
            await embed_image_chunk(opensearch, encoder, docs, plan.embed, result)
    finally:
        if owned_pool is not None and hasattr(owned_pool, 'close'):
            await owned_pool.close()
    return {
        'images': result.queued,
        'images_failed': result.failed,
        'embedded': result.detail.get('crop_written', 0),
    }


def _images_index() -> str:
    from src.config import get_curation_config

    return get_curation_config().images_index


__all__ = ['DEFAULT_TRITON_URL', 'run_embed_missing_stage', 'stage_request', 'triton_url']
