"""Run one VLM call against supplied crops and report what was asked and answered (W5).

``probe`` is the test-on-crop hook behind ``POST /prompt_packs/test``: it runs the
PRODUCTION labeler method for the call (so the parsed result is, by
construction, what the worker would get from the same reply) while
:func:`~src.services.labeling.vlm_client.capture_chat` records the exact
payload and raw reply of the upstream exchange. There are no parallel
payload builders to drift from the production ones.

Each call is capped (by the caller) at one upstream request's worth of
crops, so exactly one exchange is recorded.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from src.services.labeling.vlm_client import (
    ChatExchange,
    capture_chat,
    extract_message_content,
    extract_reasoning_content,
)
from src.services.labeling.vlm_labeler import (
    CombinedCrop,
    CombinedParseFailure,
    CombinedTransportError,
    ItemCrop,
    RegionCrop,
    VlmTransportError,
)


if TYPE_CHECKING:
    from src.services.labeling.vlm_labeler import VlmLabeler


ProbeCall = Literal['combined', 'classify', 'open_classify', 'region_verify', 'region_visible']
PROBE_CALLS: tuple[str, ...] = (
    'combined',
    'classify',
    'open_classify',
    'region_verify',
    'region_visible',
)

BBoxNorm = tuple[float, float, float, float]


@dataclass(frozen=True)
class ProbeCrop:
    """One crop of a probe. ``region_boxes`` (crop frame) are the numbered
    overlay boxes of a ``combined`` call; ignored by the others."""

    crop_id: str
    jpeg: bytes
    region_boxes: list[BBoxNorm] = field(default_factory=list)


@dataclass(frozen=True)
class ParsedEntry:
    """What the production parser made of one crop's answer. ``value`` is a
    ``VlmCombinedReply`` / ``VlmClassPrediction`` / ``VlmRegionVerdict`` /
    ``bool`` (region visibility), or ``None`` when there is no usable answer."""

    crop_id: str
    value: Any


@dataclass(frozen=True)
class ProbeResult:
    prompt_system: str | None
    prompt_user_text: str | None
    raw_reply: str | None
    reasoning: str | None
    parsed: list[ParsedEntry]
    latency_ms: float
    parse_ok: bool
    parse_error: str | None


def _first_text(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get('type') == 'text':
                return str(part.get('text', ''))
    return None


def _prompts(exchange: ChatExchange) -> tuple[str | None, str | None]:
    system: str | None = None
    user: str | None = None
    for message in exchange.payload.get('messages', []):
        if message.get('role') == 'system' and system is None:
            system = _first_text(message.get('content'))
        elif message.get('role') == 'user' and user is None:
            user = _first_text(message.get('content'))
    return system, user


async def _run_call(
    labeler: VlmLabeler,
    call: ProbeCall,
    crops: list[ProbeCrop],
    class_names: list[str] | None,
) -> tuple[list[ParsedEntry], str | None]:
    """The production method for ``call``; ``(parsed entries, parse error)``."""
    if call == 'combined':
        if len(crops) == 1:
            only = crops[0]
            try:
                reply = await labeler.label_combined(
                    only.crop_id,
                    only.jpeg,
                    class_names=class_names,
                    region_bboxes_norm=only.region_boxes,
                )
            except CombinedTransportError:
                raise
            except CombinedParseFailure as exc:
                return [ParsedEntry(only.crop_id, None)], str(exc)
            return [ParsedEntry(only.crop_id, reply)], None
        batch = await labeler.label_combined_batch(
            [
                CombinedCrop(
                    crop_id=c.crop_id,
                    jpeg_bytes=c.jpeg,
                    region_bboxes_norm=c.region_boxes,
                    classify=bool(class_names),
                )
                for c in crops
            ],
            class_names=class_names,
        )
        entries = [ParsedEntry(c.crop_id, batch.get(c.crop_id)) for c in crops]
        return entries, _missing(entries)
    if call in ('classify', 'open_classify'):
        item_crops = [ItemCrop(img_id=c.crop_id, jpeg_bytes=c.jpeg) for c in crops]
        if call == 'classify':
            predictions = await labeler.label_item_batch(item_crops, class_names or [])
        else:
            predictions = await labeler.label_or_propose_batch(item_crops, class_names or [])
        entries = [ParsedEntry(c.crop_id, p) for c, p in zip(crops, predictions, strict=True)]
        failed = [e.crop_id for e in entries if e.value.failure is not None]
        return entries, (f'no usable answer for: {", ".join(failed)}' if failed else None)
    region_crops = [RegionCrop(crop_id=c.crop_id, jpeg_bytes=c.jpeg) for c in crops]
    if call == 'region_verify':
        if len(region_crops) == 1:
            verdict = await labeler.verify_region(region_crops[0], raise_on_transport=True)
            entries = [ParsedEntry(crops[0].crop_id, verdict)]
        else:
            verdicts = {v.crop_id: v for v in await labeler.verify_region_batch(region_crops)}
            entries = [ParsedEntry(c.crop_id, verdicts.get(c.crop_id)) for c in crops]
        return entries, _missing(entries)
    visible = await labeler.region_visible_batch(region_crops)
    entries = [ParsedEntry(c.crop_id, visible.get(c.crop_id)) for c in crops]
    return entries, _missing(entries)


def _missing(entries: list[ParsedEntry]) -> str | None:
    missing = [e.crop_id for e in entries if e.value is None]
    return f'no usable answer for: {", ".join(missing)}' if missing else None


async def probe(
    labeler: VlmLabeler,
    call: ProbeCall,
    crops: list[ProbeCrop],
    *,
    class_names: list[str] | None = None,
) -> ProbeResult:
    """Run ``call`` over ``crops`` through ``labeler``.

    Raises :class:`~src.services.labeling.vlm_labeler.VlmTransportError` when
    the upstream call itself failed (no reply at all); a reply that could not
    be parsed is a result with ``parse_ok=False``, not an error.
    """
    started = time.perf_counter()
    with capture_chat() as exchanges:
        try:
            parsed, parse_error = await _run_call(labeler, call, crops, class_names)
        except CombinedTransportError as exc:
            raise VlmTransportError(str(exc)) from exc
    latency_ms = (time.perf_counter() - started) * 1000.0
    answered = [e for e in exchanges if e.response is not None]
    if not answered:
        errors = [e.error for e in exchanges if e.error]
        msg = errors[0] if errors else 'the VLM endpoint did not answer'
        raise VlmTransportError(msg)
    first = answered[0]
    system, user_text = _prompts(first)
    response = first.response or {}
    return ProbeResult(
        prompt_system=system,
        prompt_user_text=user_text,
        raw_reply=extract_message_content(response),
        reasoning=extract_reasoning_content(response) or None,
        parsed=parsed,
        latency_ms=latency_ms,
        parse_ok=parse_error is None,
        parse_error=parse_error,
    )


__all__ = [
    'PROBE_CALLS',
    'ParsedEntry',
    'ProbeCall',
    'ProbeCrop',
    'ProbeResult',
    'probe',
]
