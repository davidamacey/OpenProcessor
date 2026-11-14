"""One page of search hits as ids and a float32 matrix, and pages as one matrix.

The one decoder of a vector read: a hit carries its ``knn_vector`` through ``docvalue_fields``
as one base64 string of little-endian float32 (the 3.7 default,
https://opensearch.org/blog/retrieve-vectors-5x-faster-with-docvalue_fields-in-opensearch/).
A page is decoded in one ``np.frombuffer``, never a loop over floats.
"""

from __future__ import annotations

import base64
from typing import Any

import numpy as np


def hits_to_rows(hits: list[dict[str, Any]], field: str) -> tuple[list[str], np.ndarray | None]:
    """Ids and the float32 matrix of one page of hits, in hit order.

    A hit without the vector is skipped. A value that is not the base64 string (the ``array``
    format, a stored list) or rows of different lengths raise: a reader that guessed would
    return a different vector than the stored one.
    """
    ids: list[str] = []
    encoded: list[str] = []
    for h in hits:
        values = (h.get('fields') or {}).get(field)
        if not values:
            continue
        if not isinstance(values[0], str):
            raise TypeError(f'{field} came back as {type(values[0]).__name__}, not base64 float32')
        ids.append(h['_id'])
        encoded.append(values[0])
    if not ids:
        return ids, None
    raw = [base64.b64decode(e) for e in encoded]
    if len({len(r) for r in raw}) != 1 or len(raw[0]) % 4:
        raise ValueError(f'{field}: vectors of different or non-float32 length in one page')
    flat = np.frombuffer(b''.join(raw), dtype='<f4')
    return ids, flat.reshape(len(ids), -1).astype(np.float32)
