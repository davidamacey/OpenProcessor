"""Operator endpoints for the UMAP-backed residual clustering pipeline.

The clustering pipeline normally runs in ``transform`` mode against the
cached UMAP reducer (cheap, deterministic). After the classifier
retrains — which changes the meaning of the residual embedding field —
an operator calls ``POST {prefix}/cluster/umap/rebuild`` to refit the
reducer on the current residual pool and write the new state to both
the configured state dir's ``umap_state.joblib`` and the
``op_umap_state`` OpenSearch index.

This router is intentionally tiny — heavy logic lives in
:mod:`src.services.curation.clustering.embedding_reduce`.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from src.config import get_curation_config
from src.core.dependencies import get_opensearch
from src.core.logging import get_logger
from src.services.curation.clustering.embedding_reduce import umap_rebuild


logger = get_logger(__name__)

config = get_curation_config()

router = APIRouter(prefix=f'{config.api_prefix}/cluster', tags=[f'{config.api_tag} — Clustering'])


async def _get_os_client() -> Any:
    """Return the raw AsyncOpenSearch.

    The :class:`OpenSearchClient` wrapper doesn't expose ``.search`` /
    ``.bulk`` / ``.indices`` directly — embedding_reduce calls those on
    the raw client.
    """
    wrapper = await get_opensearch()
    return getattr(wrapper, 'client', wrapper)


@router.post('/umap/rebuild')
async def post_umap_rebuild(client: Any = Depends(_get_os_client)) -> dict[str, Any]:
    """Refit the UMAP reducer and re-cluster the residual pool.

    Returns the same envelope as :func:`cluster_residuals` —
    ``{status, n_residuals, n_clusters, n_noise, refit, ...}``.
    """
    result = await umap_rebuild(client)
    logger.info(
        'legacy_umap_rebuild_done',
        n_residuals=result.get('n_residuals'),
        n_clusters=result.get('n_clusters'),
        refit=result.get('refit'),
    )
    return result
