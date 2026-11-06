"""Candidate edges: exact kNN for a full build, OpenSearch kNN for an incremental one.

Every edge carries the exact cosine of two unit-norm vectors, computed here from the vectors
themselves, so a stored edge never depends on how an index scores (``cosinesimil`` and
``innerproduct`` spaces answer different scales). An edge is kept at or above ``min_cosine``
(the lowest cut of the profile) and ordered ``a < b``.

Docs: FAISS ``IndexFlatIP`` and ``Index.search``
(https://github.com/facebookresearch/faiss/wiki/Getting-started); OpenSearch k-NN query
(https://docs.opensearch.org/docs/latest/vector-search/vector-search-techniques/knn-core-concepts/).
The GPU build uses the same index through ``faiss.index_cpu_to_gpu`` when the faiss build has it;
the result is the same exact search either way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from src.domain.near_dup import order_pair


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from opensearchpy import AsyncOpenSearch

# Rows searched per FAISS call: bounds the (batch x n) score block in memory.
SEARCH_BATCH = 4096
KNN_BATCH = 64


def unit_rows(vectors: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """float32, L2-normalised rows (a zero row stays zero and links to nothing)."""
    mat = np.ascontiguousarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return np.where(norms > 0, mat / np.where(norms > 0, norms, 1.0), 0.0).astype(np.float32)


def _flat_index(dim: int, *, use_gpu: bool) -> Any:
    import faiss

    index = faiss.IndexFlatIP(dim)
    if use_gpu and hasattr(faiss, 'StandardGpuResources'):
        return faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
    return index


def exact_edges(
    ids: Sequence[str],
    mat: np.ndarray,
    *,
    k: int,
    min_cosine: float,
    use_gpu: bool = False,
) -> list[tuple[str, str, float]]:
    """Every pair among ``ids`` whose cosine is at or above ``min_cosine``, found through each
    row's ``k`` nearest neighbours (self excluded). ``mat`` rows follow ``ids`` and are unit norm.

    Deterministic: the search is exact and the edges are deduplicated and sorted.
    """
    if len(ids) < 2:
        return []
    index = _flat_index(mat.shape[1], use_gpu=use_gpu)
    index.add(mat)
    width = min(k + 1, len(ids))
    found: dict[tuple[str, str], float] = {}
    for start in range(0, len(ids), SEARCH_BATCH):
        scores, neighbours = index.search(mat[start : start + SEARCH_BATCH], width)
        for row, (row_scores, row_neighbours) in enumerate(zip(scores, neighbours, strict=True)):
            source = ids[start + row]
            for score, other in zip(row_scores, row_neighbours, strict=True):
                if other < 0 or score < min_cosine or ids[other] == source:
                    continue
                found[order_pair(source, ids[other])] = float(score)
    return sorted((a, b, s) for (a, b), s in found.items())


async def knn_edges(
    client: AsyncOpenSearch,
    *,
    index: str,
    field: str,
    members: Mapping[str, np.ndarray],
    k: int,
    min_cosine: float,
    id_field: str | None = None,
    extra_source: Sequence[str] = (),
    filter_for: Callable[[str], dict[str, Any] | None] | None = None,
    accept: Callable[[str, dict[str, Any]], bool] | None = None,
) -> list[tuple[str, str, float]]:
    """Edges from each member to its ``k`` nearest neighbours in the project index.

    One ``_msearch`` per ``KNN_BATCH`` members. The hit's vector is read back and the cosine
    computed locally. ``id_field`` names the stored id when the document id is not the member id.
    ``filter_for(member)`` is a k-NN pre-filter for that member's neighbours (``None``: none);
    ``accept(member, source)`` drops a neighbour after the fact (same image, other class).
    """
    source = [field, *([id_field] if id_field else []), *extra_source]
    found: dict[tuple[str, str], float] = {}
    ids = sorted(members)
    for start in range(0, len(ids), KNN_BATCH):
        batch = ids[start : start + KNN_BATCH]
        body: list[dict[str, Any]] = []
        for member in batch:
            body.append({'index': index})
            knn: dict[str, Any] = {'vector': members[member].tolist(), 'k': k + 1}
            narrowed = None if filter_for is None else filter_for(member)
            if narrowed is not None:
                knn['filter'] = narrowed
            body.append({'size': k + 1, '_source': source, 'query': {'knn': {field: knn}}})
        responses = (await client.msearch(body=body)).get('responses') or []
        for member, response in zip(batch, responses, strict=True):
            if response.get('error'):
                msg = f'near-duplicate kNN failed for {member}: {response["error"]}'
                raise RuntimeError(msg)
            for hit in response['hits']['hits']:
                other_vector = (hit.get('_source') or {}).get(field)
                other = str(hit['_source'][id_field]) if id_field else str(hit['_id'])
                if other_vector is None or other == member:
                    continue
                if accept is not None and not accept(member, hit['_source']):
                    continue
                cosine = float(unit_rows([other_vector])[0] @ members[member])
                if cosine >= min_cosine:
                    found[order_pair(member, other)] = cosine
    return sorted((a, b, s) for (a, b), s in found.items())
