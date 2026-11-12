"""Plain-language labels for an item's clarity score and size rank.

The thresholds live here, in the backend, so every client shows the same words.
The clarity bands follow the sale-quality stops the clarity slider has always
marked (1.1 and 1.4 of the Laplacian-variance ratio).
"""

from __future__ import annotations


CLARITY_CLEAR_MIN = 1.1
CLARITY_SHARP_MIN = 1.4


def clarity_label(ratio: float | None) -> str | None:
    """``soft``, ``clear`` or ``sharp`` for a blur ratio; ``None`` when unscored."""
    if ratio is None:
        return None
    if ratio < CLARITY_CLEAR_MIN:
        return 'soft'
    return 'clear' if ratio < CLARITY_SHARP_MIN else 'sharp'


def rank_label(rank: int | None) -> str | None:
    """``largest`` for rank 1, then ``2nd``, ``3rd``, ``4th``…; ``None`` when unranked."""
    if rank is None or rank < 1:
        return None
    if rank == 1:
        return 'largest'
    suffix = 'th' if 10 <= rank % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(rank % 10, 'th')
    return f'{rank}{suffix}'
