"""Region-verification operations of the VLM labeler (single and batched).

Split out of ``vlm_labeler.py``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from src.core.logging import get_logger
from src.services.labeling.vlm_client import extract_message_content, extract_reasoning_content
from src.services.labeling.vlm_labeler_core import _RESULTS_ENVELOPE, VlmLabelerCore, _b64_jpeg
from src.services.labeling.vlm_models import RegionCrop, VlmRegionVerdict, VlmTransportError
from src.services.labeling.vlm_reply_parse import (
    _align_batch_entries,
    _coerce_bool,
    _extract_region_text,
    _normalize_confidence,
    _strip_markdown_fences,
)
from src.utils.upstream_errors import describe_upstream_error


logger = get_logger(__name__)


class VerifyOps(VlmLabelerCore):
    """Is-it-a-real-region verification of sub-region crops."""

    async def verify_region(
        self, crop: RegionCrop, *, raise_on_transport: bool = False
    ) -> VlmRegionVerdict | None:
        """Verify whether a single sub-region crop is real.

        Returns ``None`` when the VLM gave no usable answer at all --
        an upstream HTTP failure, an empty reply, or a reply that never
        resolves to JSON carrying an ``is_region`` key. ``None`` is not
        evidence the region is fake; the caller must leave the crop
        pending for a retry rather than recording a rejection the VLM
        never gave.

        ``raise_on_transport=True`` raises :class:`VlmTransportError` on an
        upstream HTTP failure instead, for a caller that must tell an
        outage (retry) from a reply with no verdict (bounded retries).
        """

        b64 = _b64_jpeg(crop.jpeg_bytes)
        payload = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': self._pack.region_system},
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': self._pack.region_user},
                        {
                            'type': 'image_url',
                            'image_url': {'url': f'data:image/jpeg;base64,{b64}'},
                        },
                    ],
                },
            ],
            'temperature': 0.0,
            # Reasoning VLMs think out loud before answering, and the
            # verify-and-read prompt is longer than a verify-only ask.
            # 1024 leaves headroom for a long reasoning preamble plus
            # the ~80-token JSON reply.
            'max_tokens': 1024,
        }

        try:
            response = await self._post_chat(payload)
        except Exception as exc:
            logger.error(
                'vlm_labeler.verify_region_failed',
                crop_id=crop.crop_id,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            if raise_on_transport:
                msg = f'http error: {describe_upstream_error(exc)}'
                raise VlmTransportError(msg) from exc
            return None

        raw = _strip_markdown_fences(extract_message_content(response))
        return self._parse_region_response(raw, crop)

    async def verify_region_batch(
        self,
        crops: list[RegionCrop],
        *,
        images_per_call: int | None = None,
    ) -> list[VlmRegionVerdict]:
        """Verify a list of sub-region crops in chunks of ``max_images_per_call``.

        Packing multiple JPEGs per upstream call amortises the prompt
        prefix, the attention-warmup cost, and per-request scheduler
        overhead across several verdicts — this materially lifts
        verify throughput versus one crop per call, while staying
        inside the upstream images-per-prompt cap.

        Returns one :class:`VlmRegionVerdict` per crop the VLM actually
        answered, in input order -- fewer than ``len(crops)`` when some
        crops got no verdict (an empty/unparseable/misaligned chunk
        reply, an individual crop missing from an otherwise-aligned
        reply, or a whole-chunk upstream failure). A missing crop_id is
        not a rejection; callers must retry it, the same contract
        :py:meth:`region_visible_batch` uses for its map.
        """

        if not crops:
            return []

        per_call = images_per_call if images_per_call is not None else self.max_images_per_call
        per_call = max(1, min(per_call, self.max_images_per_call))

        chunks = [crops[i : i + per_call] for i in range(0, len(crops), per_call)]
        chunk_results_list = await asyncio.gather(
            *[self._verify_region_chunk(c) for c in chunks],
            return_exceptions=False,
        )
        results: list[VlmRegionVerdict] = []
        for cr in chunk_results_list:
            results.extend(cr)
        return results

    async def _verify_region_chunk(self, chunk: list[RegionCrop]) -> list[VlmRegionVerdict]:
        """Run one upstream verify call over up to ``max_images_per_call`` crops.

        Crops the VLM gave no usable answer for -- a whole-chunk upstream
        failure, an empty/unparseable/misaligned reply, or an individual
        crop missing from an otherwise-aligned reply -- are left out of
        the returned list entirely; absence is never recorded as a
        rejection (see :py:meth:`verify_region_batch`).
        """

        if not chunk:
            return []
        # Single-crop chunks reuse the canonical (and battle-tested)
        # single-image prompt + parser to avoid regressing the existing
        # single-call path's accuracy when callers happen to pass a
        # length-1 list.
        if len(chunk) == 1:
            verdict = await self.verify_region(chunk[0])
            return [] if verdict is None else [verdict]

        user_text = f'{self._pack.region_batch_user}\n{_RESULTS_ENVELOPE}'
        user_content: list[dict[str, Any]] = [{'type': 'text', 'text': user_text}]
        for crop in chunk:
            b64 = _b64_jpeg(crop.jpeg_bytes)
            user_content.append(
                {
                    'type': 'image_url',
                    'image_url': {'url': f'data:image/jpeg;base64,{b64}'},
                }
            )

        payload = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': self._pack.region_batch_system},
                {'role': 'user', 'content': user_content},
            ],
            'temperature': 0.0,
            # Batched verify-and-read. Per-image output schema is
            # roughly 80 tokens and the reasoning preamble scales with
            # image count. 2048 covers a worst-case 6-image batch;
            # shorter responses still stop at the real EOS.
            'max_tokens': 2048,
            # Same grammar constraint as the other batched calls: without
            # it a server-side reasoning parser can route the whole
            # answer to the reasoning channel and leave ``content``
            # empty. The json_object grammar only admits an object,
            # hence the results-envelope line appended to the prompt
            # above rather than the bare array the pack's template asks
            # for on its own.
            **self._json_mode_kwargs(),
        }

        try:
            response = await self._post_chat(payload)
        except Exception as exc:
            logger.error(
                'vlm_labeler.verify_region_chunk_failed',
                chunk_size=len(chunk),
                error=str(exc),
                error_type=type(exc).__name__,
            )
            # No answer at all: leave every crop in this chunk out of the
            # result so the caller retries, rather than synthesizing a
            # reject the VLM never gave.
            return []

        content = _strip_markdown_fences(extract_message_content(response))
        verdicts = self._parse_region_batch_response(content, chunk)
        if verdicts:
            return verdicts
        reasoning = extract_reasoning_content(response)
        if not reasoning:
            return verdicts
        from_reasoning = self._parse_region_batch_response(reasoning, chunk, log_failures=False)
        if from_reasoning:
            logger.info(
                'vlm_labeler.region_batch_reply_from_reasoning',
                chunk_size=len(chunk),
                content_preview=content[:80],
            )
            return from_reasoning
        return verdicts

    @staticmethod
    def _parse_region_batch_response(
        raw: str,
        chunk: list[RegionCrop],
        *,
        log_failures: bool = True,
    ) -> list[VlmRegionVerdict]:
        """Parse a batched verify response into the verdicts the VLM actually gave.

        No verdict at all -- an empty reply, JSON that never resolves to
        a list, an array that can't be aligned to the chunk, or an
        individual crop missing from an otherwise-aligned array -- means
        that crop is left out of the returned list. A crop with no
        answer is not evidence of a rejection; synthesizing
        ``is_region=False`` here would record a verdict the VLM never
        gave (mirrors :py:meth:`_parse_region_visible_response`'s
        empty-reply handling). ``log_failures=False`` suppresses the
        parse-failure warnings for a second attempt against the
        reasoning channel, matching :py:meth:`_parse_item_response`.

        Tolerates the same VLM quirks as :py:meth:`_parse_region_response`:
        leading reasoning prose, ``{"results":[...]}`` envelopes, and
        1-based ``img`` indices.
        """

        if not raw:
            if log_failures:
                logger.warning('vlm_labeler.region_batch_parse_empty', chunk_size=len(chunk))
            return []

        # Try the bare reply first; fall back to scanning for the first
        # balanced JSON array embedded in any preamble.
        candidates: list[str] = [raw]
        depth = 0
        start = -1
        for i, ch in enumerate(raw):
            if ch == '[':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == ']' and depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    candidates.append(raw[start : i + 1])
                    start = -1

        parsed: Any = None
        for c in candidates:
            try:
                parsed = json.loads(c)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                for key in ('results', 'predictions', 'data'):
                    if key in parsed and isinstance(parsed[key], list):
                        parsed = parsed[key]
                        break
            if isinstance(parsed, list):
                break
            parsed = None

        if not isinstance(parsed, list):
            if log_failures:
                logger.warning(
                    'vlm_labeler.region_batch_parse_failed',
                    chunk_size=len(chunk),
                    raw_preview=raw[:200],
                )
            return []

        aligned = _align_batch_entries(parsed, len(chunk))
        if aligned is None:
            if log_failures:
                logger.warning(
                    'vlm_labeler.region_batch_misaligned',
                    chunk_size=len(chunk),
                    n_entries=len(parsed),
                    raw_preview=raw[:200],
                )
            return []

        out: list[VlmRegionVerdict] = []
        for crop, entry in zip(chunk, aligned, strict=True):
            if entry is None:
                # No entry for this crop in an otherwise-aligned reply --
                # no verdict, not a reject. Leave it out of the result.
                continue
            is_region = _coerce_bool(entry.get('is_region')) is True
            confidence = _normalize_confidence(entry.get('confidence'))
            reason = str(entry.get('reason', '') or '')[:120]
            text, text_confidence = _extract_region_text(entry, is_region=is_region)
            out.append(
                VlmRegionVerdict(
                    crop_id=crop.crop_id,
                    is_region=is_region,
                    confidence=confidence,
                    reason=reason,
                    text=text,
                    text_confidence=text_confidence,
                )
            )
        return out

    @staticmethod
    def _parse_region_response(raw: str, crop: RegionCrop) -> VlmRegionVerdict | None:
        """Parse a single-region verdict, or ``None`` for no usable answer.

        Returns ``None`` -- not a low-confidence reject -- when ``raw``
        is empty or never resolves to a JSON object carrying an
        ``is_region`` key; a crop with no answer is not evidence it's a
        rejection. A VLM sometimes ignores the ``no prose`` instruction
        and emits a chain-of-thought before the JSON. We try the
        fence-stripped raw first, then fall back to extracting the first
        balanced ``{...}`` anywhere in the response so reasoning
        prefixes don't trash the verification.
        """

        if not raw:
            logger.warning('vlm_labeler.region_parse_empty', crop_id=crop.crop_id)
            return None
        candidates: list[str] = [_strip_markdown_fences(raw)]
        # Scan for the first balanced JSON object in the raw text. A
        # reasoning prefix often quotes the answer template back
        # before producing the real answer; the first balanced object
        # found this way is the actual reply.
        depth = 0
        start = -1
        for i, ch in enumerate(raw):
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}' and depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    candidates.append(raw[start : i + 1])
                    start = -1
        parsed: Any = None
        for c in candidates:
            try:
                parsed = json.loads(c)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and 'is_region' in parsed:
                break
            parsed = None
        if parsed is None:
            logger.warning(
                'vlm_labeler.region_parse_failed',
                crop_id=crop.crop_id,
                raw_preview=raw[:200],
            )
            return None
        if not isinstance(parsed, dict):
            return None

        is_region = _coerce_bool(parsed.get('is_region')) is True

        confidence = _normalize_confidence(parsed.get('confidence'))
        reason = str(parsed.get('reason', '') or '')[:120]
        text, text_confidence = _extract_region_text(parsed, is_region=is_region)
        return VlmRegionVerdict(
            crop_id=crop.crop_id,
            is_region=is_region,
            confidence=confidence,
            reason=reason,
            text=text,
            text_confidence=text_confidence,
        )
