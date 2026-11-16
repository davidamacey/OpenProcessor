"""How many decode lanes fit on a card (plan 17 section 2.3 and N2 "Profiles").

nvJPEG's device buffers of each decode lane grow to the largest photo the lane has decoded and
sit outside ``OP_FRAME_ENGINE_POOL_MIB``, so the lane count has to come from the card's memory as
well as from the cores. ``lanes_that_fit`` is the one arithmetic; the native engine's
``lane_budget.rs`` is the same function and replays the cases ``frame_engine_parity_goldens.py``
writes from this one. The installer profiles' lane counts come from it too
(``tests/test_frame_engine_profile_sizes.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


MIB = 1 << 20

#: Measured on an RTX A6000, set P 200 photos at 128 in flight, full plan, failed 0
#: (artifacts_local/speed_1113/plan17_decisions.md, "N1 on the card"): the peak of
#: ``op_frame_engine_nvjpeg_peak_bytes`` over 24 lanes while the largest photo was 26 megapixels.
MEASURED_NVJPEG_PEAK_BYTES = 5_305_587_712
MEASURED_LANES = 24
MEASURED_LARGEST_MEGAPIXELS = 26
#: Bytes one lane holds per megapixel of the largest photo it decoded (rounded up). The other
#: photo sizes (COCO, the 20 MP set) are owed to the GPU agent's next hold; replace the three
#: numbers above with the measurement that sets the figure, nothing else changes.
NVJPEG_BYTES_PER_MEGAPIXEL_PER_LANE = math.ceil(
    MEASURED_NVJPEG_PEAK_BYTES / MEASURED_LANES / MEASURED_LARGEST_MEGAPIXELS
)
#: The photo size lanes are sized for when the profile caps nothing smaller. 26 MP is the largest
#: photo measured; ``OP_FRAME_ENGINE_MAX_MEGAPIXELS`` (default 100) lower than this lowers it.
SIZED_FOR_MEGAPIXELS = 26.0
#: Left free for what is not counted (driver work areas, the model server's own growth), percent
#: of the memory left after the pool.
HEADROOM_PERCENT = 10
MIN_LANES = 1
MAX_LANES = 256  # the setting's range


@dataclass(frozen=True)
class LaneFit:
    """The lane count and whether even the minimum fits."""

    lanes: int
    fits: bool


def lanes_that_fit(
    free_bytes: int,
    pool_bytes: int,
    largest_megapixels: float,
    bytes_per_megapixel: int = NVJPEG_BYTES_PER_MEGAPIXEL_PER_LANE,
    headroom_percent: int = HEADROOM_PERCENT,
) -> LaneFit:
    """Lanes whose grown buffers fit in what the card has free once the pool cap is set aside.

    ``free_bytes`` is the card's free memory after the context and the region are taken. Never
    more than fit, never fewer than ``MIN_LANES`` (``fits`` is false then), never more than
    ``MAX_LANES``."""
    usable = max(free_bytes - pool_bytes, 0)
    usable = usable * (100 - headroom_percent) // 100
    per_lane = max(math.ceil(largest_megapixels * bytes_per_megapixel), 1)
    fit = usable // per_lane
    return LaneFit(lanes=min(max(fit, MIN_LANES), MAX_LANES), fits=fit >= MIN_LANES)


def sized_megapixels(max_megapixels: float) -> float:
    """The photo size to size the lanes for under ``OP_FRAME_ENGINE_MAX_MEGAPIXELS``."""
    return min(max_megapixels, SIZED_FOR_MEGAPIXELS)


#: The most lanes an installer profile writes: the Python engine's default, measured.
PROFILE_LANE_CAP = 16
CONTEXT_MIB = 1024


def profile_lanes(
    decode_on: bool, card_free_mib: int, models_mib: int, region_mib: int, pool_mib: int
) -> int:
    """``frame_engine_lanes`` of an installer profile: one lane where the GPU decode is off, else
    what fits on the profile's smallest card beside its models, at most ``PROFILE_LANE_CAP``."""
    if not decode_on:
        return MIN_LANES
    free = (card_free_mib - models_mib - CONTEXT_MIB - region_mib) * MIB
    fit = lanes_that_fit(free, pool_mib * MIB, SIZED_FOR_MEGAPIXELS)
    return min(fit.lanes, PROFILE_LANE_CAP)
