"""What one answered VLM call cost, read off the response (pure).

An OpenAI-shaped ``/chat/completions`` response carries ``usage`` with
``prompt_tokens`` and ``completion_tokens``. A response without it is counted as
``usage_missing`` and charged at the reserved upper bound by the budget (research
auto-label plan, section 5.5), never as free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VlmUsage:
    """Token counts of one call; ``present`` is false when the endpoint sent no usage object."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    present: bool = False


NO_USAGE = VlmUsage()


def _count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def usage_from_response(response: dict[str, Any] | None) -> VlmUsage:
    """The usage of ``response``; ``NO_USAGE`` unless both counts are non-negative integers."""
    raw = response.get('usage') if isinstance(response, dict) else None
    if not isinstance(raw, dict):
        return NO_USAGE
    prompt, completion = _count(raw.get('prompt_tokens')), _count(raw.get('completion_tokens'))
    if prompt is None or completion is None:
        return NO_USAGE
    return VlmUsage(prompt_tokens=prompt, completion_tokens=completion, present=True)


def crops_in_payload(payload: dict[str, Any]) -> int:
    """Images sent in one chat payload: the unit of the crop budget."""
    total = 0
    for message in payload.get('messages') or ():
        content = message.get('content') if isinstance(message, dict) else None
        if isinstance(content, list):
            total += sum(
                1 for part in content if isinstance(part, dict) and part.get('type') == 'image_url'
            )
    return total
