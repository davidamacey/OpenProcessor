"""Combined class + region-verify + region-text operations of the VLM labeler.

Split out of ``vlm_labeler.py``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import TYPE_CHECKING, Any

from src.core.logging import get_logger
from src.services.labeling.region_overlay import draw_region_overlay, render_region_block
from src.services.labeling.vlm_client import extract_message_content
from src.services.labeling.vlm_labeler_core import VlmLabelerCore, _b64_jpeg
from src.services.labeling.vlm_models import (
    CombinedCrop,
    CombinedParseFailure,
    CombinedTransportError,
    VlmCombinedReply,
)
from src.services.labeling.vlm_reply_parse import (
    _align_batch_entries,
    _combined_reply_from_entry,
    _strip_markdown_fences,
)
from src.utils.upstream_errors import describe_upstream_error


if TYPE_CHECKING:
    from src.config import RegionFields


logger = get_logger(__name__)


class CombinedOps(VlmLabelerCore):
    """One-call class + region labeling, single and batched."""

    async def label_combined(
        self,
        img_id: str,
        jpeg_bytes: bytes,
        *,
        class_names: list[str] | None = None,
        region_bboxes_norm: list[tuple[float, float, float, float]] | None = None,
        draw_overlay: bool = True,
    ) -> VlmCombinedReply:
        """One VLM call returns class + region-verify + region-text.

        Args:
            img_id: Crop identifier (echoed back on the reply).
            jpeg_bytes: Crop JPEG bytes.
            class_names: Class-name slice to classify against. Pass
                None / [] when the caller only wants the region-side
                answers; ``class_id`` returns None.
            region_bboxes_norm: This item candidate region bboxes in
                normalized crop coords ``[x1, y1, x2, y2]`` (W8: always a
                list, possibly empty/None). Empty/None when no candidate
                exists — the VLM still answers ``region_visible``.
            draw_overlay: If True (default), draw the numbered region
                overlay on the crop bytes before encoding so the VLM
                reasons about each box visually. Falls back to
                coords-in-prompt if drawing fails or is disabled.

        Returns:
            :class:`VlmCombinedReply` with class + region fields filled
            per the cohort.

        Raises:
            CombinedParseFailure: response unparseable. Caller falls
                back to the separate-call paths.
        """
        boxes = region_bboxes_norm or []
        bytes_to_send = jpeg_bytes
        overlay_drawn = False
        if draw_overlay and boxes:
            drew = draw_region_overlay(jpeg_bytes, boxes)
            if drew is not None:
                bytes_to_send = drew
                overlay_drawn = True

        if class_names:
            class_block = (
                'Identify the item class. Choose ONE class_id from: '
                + ', '.join(f'{i}={name}' for i, name in enumerate(class_names))
                + '. '
            )
        else:
            class_block = "Don't classify (the caller already has a class). Set class_id=null. "

        region_block = render_region_block(boxes, overlay_drawn=overlay_drawn)

        user_text = self._pack.combined_user_template.format(
            class_block=class_block, region_block=region_block
        )
        b64 = _b64_jpeg(bytes_to_send)
        user_content: list[dict[str, Any]] = [
            {'type': 'text', 'text': user_text},
            {
                'type': 'image_url',
                'image_url': {'url': f'data:image/jpeg;base64,{b64}'},
            },
        ]

        payload: dict[str, Any] = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': self._pack.combined_system},
                {'role': 'user', 'content': user_content},
            ],
            'temperature': 0.0,
            'max_tokens': 256,
            **self._json_mode_kwargs(),
        }

        try:
            response = await self._post_chat(payload)
        except Exception as exc:
            logger.warning(
                'vlm_labeler.label_combined_http_failed',
                img_id=img_id,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            raise CombinedTransportError(f'http error: {describe_upstream_error(exc)}') from exc

        raw = _strip_markdown_fences(extract_message_content(response))
        if not raw:
            logger.info('vlm_labeler.combined_parse_failure', img_id=img_id, reason='empty')
            raise CombinedParseFailure('empty response')
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.info(
                'vlm_labeler.combined_parse_failure',
                img_id=img_id,
                raw_excerpt=raw[:200],
            )
            raise CombinedParseFailure(f'invalid json: {exc}') from exc

        if not isinstance(parsed, dict):
            logger.info('vlm_labeler.combined_parse_failure', img_id=img_id, reason='not-an-object')
            raise CombinedParseFailure('response is not a json object')

        try:
            return _combined_reply_from_entry(
                parsed,
                img_id=img_id,
                fields=self._fields,
                class_names=class_names,
                n_boxes=len(boxes),
            )
        except (TypeError, ValueError) as exc:
            logger.info(
                'vlm_labeler.combined_parse_failure',
                img_id=img_id,
                reason='type-coercion',
                error=str(exc),
            )
            raise CombinedParseFailure(f'type coercion: {exc}') from exc

    async def label_combined_batch(
        self,
        crops: list[CombinedCrop],
        *,
        class_names: list[str] | None = None,
        images_per_call: int | None = None,
        draw_overlay: bool = True,
    ) -> dict[str, VlmCombinedReply | None]:
        """Batched combined call: class + region-verify + region-text for many crops.

        Packs ``images_per_call`` (default ``max_images_per_call``) crops
        per upstream call. Each crop carries an optional candidate
        region bbox which is drawn as a colored overlay on the JPEG
        before encoding so the VLM can reason about it visually.

        Returns ``{crop_id: VlmCombinedReply | None}``. A ``None`` value
        means the VLM answered but the per-crop entry could not be parsed
        (missing in response, bad JSON) — a reply without a verdict.

        Raises:
            CombinedTransportError: an upstream call failed (HTTP /
                transport) -- no reply at all, for the whole batch.
        """

        if not crops:
            return {}

        per_call = images_per_call if images_per_call is not None else self.max_images_per_call
        per_call = max(1, min(per_call, self.max_images_per_call))

        chunks = [crops[i : i + per_call] for i in range(0, len(crops), per_call)]
        chunk_results = await asyncio.gather(
            *[
                self._label_combined_chunk(
                    c,
                    class_names=class_names,
                    draw_overlay=draw_overlay,
                )
                for c in chunks
            ],
            return_exceptions=False,
        )
        merged: dict[str, VlmCombinedReply | None] = {}
        for d in chunk_results:
            merged.update(d)
        return merged

    async def _label_combined_chunk(
        self,
        chunk: list[CombinedCrop],
        *,
        class_names: list[str] | None,
        draw_overlay: bool,
    ) -> dict[str, VlmCombinedReply | None]:
        """Run one upstream combined call over a chunk of crops."""

        if not chunk:
            return {}
        # Length-1 chunks reuse the single-image path so we don't pay the
        # numbered-image prompt overhead for what is functionally just
        # ``label_combined``.
        if len(chunk) == 1:
            crop = chunk[0]
            try:
                reply = await self.label_combined(
                    img_id=crop.crop_id,
                    jpeg_bytes=crop.jpeg_bytes,
                    class_names=class_names if crop.classify else None,
                    region_bboxes_norm=crop.region_bboxes_norm,
                    draw_overlay=draw_overlay,
                )
                return {crop.crop_id: reply}
            except CombinedTransportError:
                raise
            except CombinedParseFailure:
                return {crop.crop_id: None}

        any_classify = any(c.classify for c in chunk)
        header = self._pack.combined_batch_rules
        if any_classify and class_names:
            catalog = (
                'Class catalog (use ``class_id`` to refer to entries by index): '
                + ', '.join(f'{i}={name}' for i, name in enumerate(class_names))
                + '.\n'
            )
            header = catalog + header

        user_content: list[dict[str, Any]] = [{'type': 'text', 'text': header}]
        for i, crop in enumerate(chunk, start=1):
            boxes = crop.region_bboxes_norm
            bytes_to_send = crop.jpeg_bytes
            overlay_drawn = False
            if draw_overlay and boxes:
                drew = draw_region_overlay(crop.jpeg_bytes, boxes)
                if drew is not None:
                    bytes_to_send = drew
                    overlay_drawn = True

            if crop.classify:
                directive_class = 'classify the item (choose one ``class_id`` from the catalog)'
            else:
                directive_class = 'skip classification (set ``class_id``=null)'

            directive_region = render_region_block(boxes, overlay_drawn=overlay_drawn)
            directive = f'Image {i}: {directive_class}. {directive_region}'

            b64 = _b64_jpeg(bytes_to_send)
            user_content.append({'type': 'text', 'text': directive})
            user_content.append(
                {
                    'type': 'image_url',
                    'image_url': {'url': f'data:image/jpeg;base64,{b64}'},
                }
            )

        payload = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': self._pack.combined_batch_system},
                {'role': 'user', 'content': user_content},
            ],
            'temperature': 0.0,
            # Combined prompt is bigger than batched verify (per-image
            # directive + class catalog + 8-field JSON schema); some
            # VLMs also emit 1000-1500 tokens of chain-of-thought
            # preamble on multi-image prompts before the JSON. 6144
            # leaves comfortable headroom for a 6-image batch without
            # truncating the closing bracket.
            'max_tokens': 6144,
            # Grammar-constrain the output to a valid JSON object when
            # supported by the upstream server. Without it, a reasoning
            # model can consume the entire turn in the reasoning
            # channel and emit zero visible content. vLLM's json_object
            # grammar only covers objects (not bare arrays), which is
            # why ``combined_batch_system`` asks for
            # ``{"results": [...]}`` rather than a top-level array.
            **self._json_mode_kwargs(),
        }

        try:
            response = await self._post_chat(payload)
        except Exception as exc:
            logger.error(
                'vlm_labeler.label_combined_chunk_failed',
                chunk_size=len(chunk),
                error=str(exc),
                error_type=type(exc).__name__,
            )
            # No reply at all: raise, so the caller can tell an outage
            # (keep retrying) from a reply that gave no verdict.
            msg = f'http error: {describe_upstream_error(exc)}'
            raise CombinedTransportError(msg) from exc

        raw = _strip_markdown_fences(extract_message_content(response))
        finish_reason = ''
        with contextlib.suppress(KeyError, TypeError, IndexError):
            finish_reason = str(response['choices'][0].get('finish_reason') or '')
        if not raw or finish_reason == 'length':
            # Diagnostic breadcrumb when the VLM either returned empty
            # or was truncated by max_tokens.
            logger.warning(
                'vlm_labeler.combined_batch_response_truncated_or_empty',
                chunk_size=len(chunk),
                finish_reason=finish_reason,
                raw_len=len(raw),
                raw_preview=raw[:300],
            )
        return self._parse_combined_batch_response(
            raw, chunk, self._fields, class_names=class_names
        )

    @staticmethod
    def _parse_combined_batch_response(
        raw: str,
        chunk: list[CombinedCrop],
        fields: RegionFields,
        *,
        class_names: list[str] | None = None,
    ) -> dict[str, VlmCombinedReply | None]:
        """Parse a batched combined response into ``{crop_id: reply | None}``.

        Mirrors the tolerant pattern of :py:meth:`_parse_region_batch_response`:
        accepts a bare JSON array OR an array embedded in reasoning prose,
        unwraps ``{"results": [...]}`` envelopes, and tolerates positional
        entries that omit the ``img`` index. Per-crop parse failures (or
        an entirely empty response) return ``None`` for the affected crops
        so the caller can leave them in pending instead of stamping a
        terminal status — same policy as ``label_combined`` raising
        :class:`CombinedParseFailure`.
        """

        if not raw:
            logger.warning('vlm_labeler.combined_batch_parse_empty', chunk_size=len(chunk))
            return {c.crop_id: None for c in chunk}

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
                'vlm_labeler.combined_batch_parse_failed',
                chunk_size=len(chunk),
                raw_len=len(raw),
                raw_preview=raw[:400],
            )
            return {c.crop_id: None for c in chunk}

        aligned = _align_batch_entries(parsed, len(chunk))
        if aligned is None:
            logger.warning(
                'vlm_labeler.combined_batch_misaligned',
                chunk_size=len(chunk),
                n_entries=len(parsed),
                raw_preview=raw[:400],
            )
            return {c.crop_id: None for c in chunk}

        out: dict[str, VlmCombinedReply | None] = {}
        for crop, entry in zip(chunk, aligned, strict=True):
            if entry is None:
                out[crop.crop_id] = None
                continue
            try:
                out[crop.crop_id] = _combined_reply_from_entry(
                    entry,
                    img_id=crop.crop_id,
                    fields=fields,
                    class_names=class_names if crop.classify else None,
                    n_boxes=len(crop.region_bboxes_norm),
                )
            except (TypeError, ValueError) as exc:
                logger.warning(
                    'vlm_labeler.combined_batch_entry_invalid',
                    crop_id=crop.crop_id,
                    error=str(exc),
                    entry_preview=json.dumps(entry, default=str)[:300],
                )
                out[crop.crop_id] = None
        return out
