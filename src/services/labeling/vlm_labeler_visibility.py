"""Region-visibility operations of the VLM labeler (the cheap yes/no pre-checks).

Split out of ``vlm_labeler.py``.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from src.core.logging import get_logger
from src.services.labeling.vlm_client import extract_message_content
from src.services.labeling.vlm_labeler_core import VlmLabelerCore, _b64_jpeg
from src.services.labeling.vlm_reply_parse import (
    _align_batch_entries,
    _coerce_bool,
    _strip_markdown_fences,
)


if TYPE_CHECKING:
    from src.config import RegionFields
    from src.services.labeling.vlm_models import RegionCrop


logger = get_logger(__name__)


class VisibilityOps(VlmLabelerCore):
    """Yes/no visibility questions asked before the expensive region stages."""

    async def prompt_visible(self, jpeg: bytes, prompt: str) -> bool | None:
        """Ask whether something described by ``prompt`` is visible in one
        image (the segmenter gate's tier-2 pre-check).

        ``True`` / ``False`` is the model's answer. ``None`` is NO answer (a
        failed call, an unreadable or non-boolean reply): the caller must not
        treat it as "not visible".
        """
        payload = {
            'model': self.model,
            'messages': [
                {
                    'role': 'system',
                    'content': 'You answer a yes/no question about an image with JSON only.',
                },
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': (
                                f'Is there a {prompt} visible in this image, even partly or '
                                'small? Reply with {"visible": true} or {"visible": false}.'
                            ),
                        },
                        {
                            'type': 'image_url',
                            'image_url': {'url': f'data:image/jpeg;base64,{_b64_jpeg(jpeg)}'},
                        },
                    ],
                },
            ],
            'temperature': 0.0,
            'max_tokens': 1024,
            **self._json_mode_kwargs(),
        }
        try:
            response = await self._post_chat(payload)
            parsed = json.loads(_strip_markdown_fences(extract_message_content(response)))
        except Exception as exc:
            logger.warning(
                'vlm_labeler.prompt_visible_failed', error=str(exc), error_type=type(exc).__name__
            )
            return None
        return _coerce_bool(parsed.get('visible')) if isinstance(parsed, dict) else None

    async def region_visible_batch(
        self,
        crops: list[RegionCrop],
        *,
        images_per_call: int | None = None,
    ) -> dict[str, bool]:
        """Pre-filter crops by asking the VLM whether a sub-region is visible.

        Why this exists
        ---------------
        A full region-of-interest detector (e.g. an interactive
        segmentation model) is often the throughput bottleneck. A
        non-trivial fraction of item crops have no visible sub-region
        at all. Asking a one-bit yes/no question up front — packed
        several crops per call — is far cheaper than letting those
        crops walk the full detect → verify pipeline only to be
        discarded downstream.

        Returns ``{crop_id: bool}`` — ``True`` means the sub-region
        appears visible (continue to the detector), ``False`` means
        skip the detector and write a terminal "not visible" status
        directly. On an RPC failure or a garbled entry for a crop the
        verdict defaults to ``True`` so we never silently drop a crop
        that might have a real sub-region — the existing detector path
        remains the safety net. A chunk the VLM answered with nothing is
        absent from the result: no verdict, retry later.
        """

        if not crops:
            return {}

        per_call = images_per_call if images_per_call is not None else self.max_images_per_call
        per_call = max(1, min(per_call, self.max_images_per_call))

        chunks = [crops[i : i + per_call] for i in range(0, len(crops), per_call)]
        chunk_results = await asyncio.gather(
            *[self._region_visible_chunk(c) for c in chunks],
            return_exceptions=False,
        )
        merged: dict[str, bool] = {}
        for d in chunk_results:
            merged.update(d)
        return merged

    async def _region_visible_chunk(self, chunk: list[RegionCrop]) -> dict[str, bool]:
        """Run one yes/no upstream call over up to ``max_images_per_call`` crops."""

        if not chunk:
            return {}

        user_content: list[dict[str, Any]] = [
            {'type': 'text', 'text': self._pack.region_visible_user}
        ]
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
                {'role': 'system', 'content': self._pack.region_visible_system},
                {'role': 'user', 'content': user_content},
            ],
            'temperature': 0.0,
            # Reply is a tight array of ~12 tokens per image, but a
            # reasoning model can emit a 200-400 token chain-of-thought
            # preamble on multi-image prompts. 1024 leaves comfortable
            # headroom; the actual final JSON is still ~12 tokens per
            # image so the wire cost is bounded by what comes after the
            # reasoning section.
            'max_tokens': 1024,
            # Grammar-constrain to a JSON object for the same reason as
            # ``label_combined_batch``; the prompt asks for
            # ``{"results": [...]}`` since the json_object grammar only
            # covers top-level objects.
            **self._json_mode_kwargs(),
        }

        try:
            response = await self._post_chat(payload)
        except Exception as exc:
            logger.error(
                'vlm_labeler.region_visible_chunk_failed',
                chunk_size=len(chunk),
                error=str(exc),
                error_type=type(exc).__name__,
            )
            # Fail-open: assume the sub-region is visible so the
            # detector still gets a crack at it. Costs a wasted
            # detector call we could have skipped but never silently
            # drops a real region.
            return {c.crop_id: True for c in chunk}

        raw = _strip_markdown_fences(extract_message_content(response))
        return self._parse_region_visible_response(raw, chunk, self._fields)

    @staticmethod
    def _parse_region_visible_response(
        raw: str,
        chunk: list[RegionCrop],
        fields: RegionFields,
    ) -> dict[str, bool]:
        """Parse the visibility batch reply into ``{crop_id: bool}``.

        No verdict on an empty response: an empty raw from the upstream
        VLM almost always means the slot timed out or the request
        aborted under heavy concurrent load. The chunk's crops are left
        out of the result, and the caller retries them later -- neither
        sent through the detector (the throughput cost that once made
        this fail-closed) nor stamped with a terminal "no region
        visible" the VLM never said.

        Per-entry parse failures (missing img index, unparseable
        verdict) remain fail-open because at that point we have
        evidence the VLM did respond — the response was just
        malformed for this image. Fail-open keeps recall intact for
        genuine model-confused cases.
        """

        if not raw:
            # No answer is no verdict: the caller retries these crops. It
            # must not read as "no region visible" -- that is a terminal
            # write a slot timeout would otherwise stamp on a whole chunk.
            logger.warning('vlm_labeler.region_visible_parse_empty', chunk_size=len(chunk))
            return {}

        # Default fail-open per-crop verdict (VLM responded but maybe
        # garbled a few entries — keep recall on those).
        out: dict[str, bool] = {c.crop_id: True for c in chunk}

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
            logger.warning(
                'vlm_labeler.region_visible_parse_failed',
                chunk_size=len(chunk),
                raw_preview=raw[:200],
            )
            return out

        aligned = _align_batch_entries(parsed, len(chunk))
        if aligned is None:
            # Can't tell which verdict is whose: keep the fail-open default.
            logger.warning(
                'vlm_labeler.region_visible_misaligned',
                chunk_size=len(chunk),
                n_entries=len(parsed),
            )
            return out

        for crop, entry in zip(chunk, aligned, strict=True):
            if entry is None:
                # Fail-open: leave the default True verdict in place.
                continue
            visible_raw = entry.get('visible')
            if visible_raw is None:
                # Tolerate alternate keys callers might emit.
                visible_raw = entry.get('is_region') or entry.get(fields.visible)
            if isinstance(visible_raw, bool):
                out[crop.crop_id] = visible_raw
            elif isinstance(visible_raw, str):
                v = visible_raw.strip().lower()
                if v in ('true', 'yes', 'y', '1', 'visible'):
                    out[crop.crop_id] = True
                elif v in ('false', 'no', 'n', '0', 'not_visible', 'hidden'):
                    out[crop.crop_id] = False
                # else leave as default True (fail-open).
            # Non-bool, non-str → leave as default True.
        return out
