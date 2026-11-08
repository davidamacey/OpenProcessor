"""The ``embedding_reindex`` task: binds the re-index logic to the job runtime (embeddings plan 5.5).

The index and encoder adapters are client factories the worker image registers; until they are
wired a delivery fails on the missing factory (fail closed) instead of guessing a backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.tasks.runtime.register import register_job


if TYPE_CHECKING:
    from src.tasks.runtime.context import JobContext


async def embedding_reindex(ctx: JobContext) -> Any:
    from src.services.embedding_spaces.reindex import run_reindex

    return await run_reindex(ctx, await ctx.client('reindex_index'), await ctx.client('encoder'))


register_job('embedding_reindex', embedding_reindex)
