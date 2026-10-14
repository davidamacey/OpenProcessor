"""Vector conversion for JSON documents."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np


def float_list(vector: Sequence[float] | np.ndarray) -> list[float]:
    """``vector`` as a list of Python floats, converted in one C call.

    ``list(ndarray)`` makes a list of numpy scalars that the JSON encoder then
    converts one by one through its ``default`` hook (about 1,000 Python calls
    per embedding); ``ndarray.tolist()`` yields the same float values directly.
    Anything that is not an array is copied as it is.
    """
    tolist = getattr(vector, 'tolist', None)
    return tolist() if tolist is not None else list(vector)
