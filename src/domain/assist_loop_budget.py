"""Reservation arithmetic of the assist loop's budgets (pure; the SQL is ``db/repos/assist_loop.py``).

One upstream call reserves five units before it is sent (research auto-label plan, section 5.5):
the images in it (crops), one request, completion tokens as the call's exact ``max_tokens`` (an
upper bound the endpoint cannot exceed) and prompt tokens as an estimate. A settled call spends
what the endpoint reported in ``usage``; a call with no usage object is charged what was reserved
(fail closed), and the loop counts it as ``usage_missing``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from src.domain.vlm_usage import VlmUsage

POOLS: tuple[str, ...] = ('vlm', 'audit')
UNITS: tuple[str, ...] = ('crops', 'requests', 'prompt_tokens', 'completion_tokens', 'wall_s')
PROMPT_HISTORY_CALLS = 20
PROMPT_MARGIN = 1.2
# Prompt tokens one image costs when no call has been measured yet; a deliberately high figure,
# so the first reservations of a run err towards refusing, not towards overshooting.
FIRST_CALL_TOKENS_PER_IMAGE = 1500


@dataclass(frozen=True)
class CallUnits:
    """What one call reserves, by unit (``wall_s`` is charged per slice, never per call)."""

    crops: int
    requests: int
    prompt_tokens: int
    completion_tokens: int

    def as_amounts(self) -> dict[str, int]:
        return {
            'crops': self.crops,
            'requests': self.requests,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
        }


def estimate_prompt_tokens(
    recent_per_image: Sequence[float], images: int, prompt_length_tokens: int
) -> int:
    """The last :data:`PROMPT_HISTORY_CALLS` calls' mean prompt tokens per image times ``images``,
    plus the text prompt's length, plus 20 percent."""
    recent = recent_per_image[-PROMPT_HISTORY_CALLS:]
    per_image = sum(recent) / len(recent) if recent else FIRST_CALL_TOKENS_PER_IMAGE
    return math.ceil((per_image * images + prompt_length_tokens) * PROMPT_MARGIN)


def call_units(
    *,
    images: int,
    max_tokens: int,
    prompt_length_tokens: int,
    recent_per_image: Sequence[float] = (),
) -> CallUnits:
    if images < 1 or max_tokens < 1:
        raise ValueError('a call has at least one image and a positive max_tokens')
    return CallUnits(
        crops=images,
        requests=1,
        prompt_tokens=estimate_prompt_tokens(recent_per_image, images, prompt_length_tokens),
        completion_tokens=max_tokens,
    )


def settled_amounts(reserved: Mapping[str, int], usage: VlmUsage) -> dict[str, int]:
    """What a settled call spends, by unit. Crops and requests are always the reservation (they
    were sent). Tokens come from ``usage``; completion tokens are clamped to the reservation,
    which ``max_tokens`` bounds. Without usage the whole reservation is spent."""
    spent = {unit: reserved.get(unit, 0) for unit in ('crops', 'requests')}
    if usage.present:
        spent['prompt_tokens'] = usage.prompt_tokens
        spent['completion_tokens'] = min(usage.completion_tokens, reserved['completion_tokens'])
    else:
        spent['prompt_tokens'] = reserved['prompt_tokens']
        spent['completion_tokens'] = reserved['completion_tokens']
    return spent
