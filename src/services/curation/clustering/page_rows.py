"""One page of search hits as ids and a float32 matrix, and pages as one matrix.

Shared by the residual fetches and the incremental assign pass, which read the same
vector field from the same kind of hit.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def hits_to_rows(hits: list[dict[str, Any]], field: str) -> tuple[list[str], np.ndarray | None]:
    """Ids and the float32 matrix of one page of hits, in hit order.

    A hit without the vector is skipped. The whole page is converted in one call (a Python
    list of 1,024 floats per document is the input either way), which is cheaper than a
    ``np.asarray`` per document and a ``vstack`` after; the values are the same.
    """
    ids: list[str] = []
    vectors: list[list[float]] = []
    for h in hits:
        emb = (h.get('_source') or {}).get(field)
        if emb is None:
            continue
        ids.append(h['_id'])
        vectors.append(emb)
    if not ids:
        return ids, None
    return ids, np.array(vectors, dtype=np.float32)


def stack_pages(pages: list[np.ndarray]) -> np.ndarray:
    return pages[0] if len(pages) == 1 else np.concatenate(pages)
