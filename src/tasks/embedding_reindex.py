"""The ``embedding_reindex`` task: binds the re-index logic to the job runtime (embeddings plan 5.5).

The index and the encoder are client factories; the wrapper closes each client it built. Both
reach OpenSearch through ``make_script_opensearch`` (the project guard), and the encoder owns the
Triton pool the shared ``PEEncoder`` sends its batches through.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.services.curation.job_binding import opensearch_client
from src.tasks.runtime.register import register_job


if TYPE_CHECKING:
    from src.tasks.runtime.context import JobContext


async def _reindex_index() -> Any:
    from src.services.embedding_spaces.reindex_opensearch import OpenSearchReindexIndex

    return OpenSearchReindexIndex(await opensearch_client(60))


async def _encoder() -> Any:
    from src.clients.pe_encoder import PEEncoder
    from src.clients.triton_pool_core import AsyncTritonPool
    from src.config.settings import get_settings
    from src.services.embedding_spaces.reindex_encoder import ReindexEncoder

    pool = AsyncTritonPool(
        url=get_settings().triton_url, pool_size=4, max_concurrent=64, verbose=False
    )
    await pool.initialize()
    client = await opensearch_client(60)
    return ReindexEncoder(client, PEEncoder(triton_pool=pool), owned=[client, pool])


async def embedding_reindex(ctx: JobContext) -> Any:
    from src.services.embedding_spaces.reindex import run_reindex

    return await run_reindex(ctx, await ctx.client('reindex_index'), await ctx.client('encoder'))


register_job(
    'embedding_reindex',
    embedding_reindex,
    client_factories={'reindex_index': _reindex_index, 'encoder': _encoder},
)
