"""Representative sampling for assist runs: pure, stdlib only, and independent
of how items were clustered. A scope is a set of ids (optionally grouped into
strata such as sub-clusters); the sample is reproducible from the ids, the
size and the seed."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

SAMPLE_FLOOR = 8
"""Below this a scope is sampled whole."""
SAMPLE_SQRT_FACTOR = 2.0
SAMPLE_DEFAULT_CAP = 64
SAMPLE_HARD_MAX = 256
MAX_ASSIST_ITEMS = 5000
"""VLM items per assist run: the sample size, or the scope size for mode ``all``."""


def representative_size(n: int, *, size: int | None = None) -> int:
    """How many of ``n`` scope members to send: ``size`` capped by ``n``, else
    ``max(8, min(64, ceil(2 * sqrt(n))))`` capped by ``n``."""
    if size is not None:
        if not 1 <= size <= SAMPLE_HARD_MAX:
            msg = f'size must be 1..{SAMPLE_HARD_MAX}'
            raise ValueError(msg)
        return min(n, size)
    scaled = math.ceil(SAMPLE_SQRT_FACTOR * math.sqrt(n))
    return min(n, max(SAMPLE_FLOOR, min(SAMPLE_DEFAULT_CAP, scaled)))


def _allocate(sizes: Mapping[str, int], k: int) -> dict[str, int]:
    """Picks per stratum summing to ``k`` (``k <= sum(sizes)``): one each when
    ``k`` covers every stratum, the rest proportional to size by largest
    remainder, never above a stratum's size. With fewer picks than strata the
    ``k`` largest strata (ties by key) get one each."""
    if k < len(sizes):
        ranked = sorted(sizes, key=lambda key: (-sizes[key], key))
        return dict.fromkeys(ranked[:k], 1)
    alloc = dict.fromkeys(sizes, 1)
    remaining = k - len(sizes)
    while remaining > 0:
        open_keys = [key for key in sorted(sizes) if alloc[key] < sizes[key]]
        if not open_keys:
            break
        weight = sum(sizes[key] for key in open_keys)
        shares = {key: remaining * sizes[key] / weight for key in open_keys}
        given = 0
        for key in open_keys:
            take = min(int(shares[key]), sizes[key] - alloc[key])
            alloc[key] += take
            given += take
        leftover = remaining - given
        by_fraction = sorted(open_keys, key=lambda key: (-(shares[key] - int(shares[key])), key))
        for key in by_fraction:
            if leftover == 0:
                break
            if alloc[key] < sizes[key]:
                alloc[key] += 1
                given += 1
                leftover -= 1
        remaining -= given
    return alloc


def stratified_sample(
    ids_by_stratum: Mapping[str, Sequence[str]], k: int, *, seed: int
) -> list[str]:
    """``k`` ids drawn across the strata (see :func:`_allocate`). Within a
    stratum the ids are sorted, then drawn with ``Random(f'{seed}:{stratum}')``,
    so the result depends only on the ids, ``k`` and ``seed``. Returned sorted."""
    sizes = {key: len(ids) for key, ids in ids_by_stratum.items() if ids}
    total = sum(sizes.values())
    if k <= 0 or total == 0:
        return []
    alloc = _allocate(sizes, min(k, total))
    picked: list[str] = []
    for key, count in alloc.items():
        pool = sorted(ids_by_stratum[key])
        picked.extend(random.Random(f'{seed}:{key}').sample(pool, count))
    return sorted(picked)
