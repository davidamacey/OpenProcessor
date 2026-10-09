"""Class-labeling operations of the VLM labeler (closed- and open-vocabulary).

Split out of ``vlm_labeler.py``.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from src.core.logging import get_logger
from src.services.labeling.vlm_client import extract_message_content, extract_reasoning_content
from src.services.labeling.vlm_labeler_core import _RESULTS_ENVELOPE, VlmLabelerCore, _b64_jpeg
from src.services.labeling.vlm_models import ItemCrop, VlmClassPrediction
from src.services.labeling.vlm_prompts import proposal_denied
from src.services.labeling.vlm_reply_parse import (
    _class_reply_entries,
    _normalize_confidence,
    _request_failed,
    _strip_markdown_fences,
)


if TYPE_CHECKING:
    from src.config import RegionFields


logger = get_logger(__name__)


class ClassifyOps(VlmLabelerCore):
    """Batch classification of item crops against a class list."""

    async def label_item_batch(
        self,
        crops: list[ItemCrop],
        class_names: list[str],
    ) -> list[VlmClassPrediction]:
        """Label up to ``len(crops)`` item crops, chunked at ``max_images_per_call``.

        Returns one :class:`VlmClassPrediction` per input crop, in the
        same order. On unrecoverable parse errors a low-confidence empty
        prediction is returned for the affected crop so downstream code
        can still funnel it into the human-review queue.
        """

        if not crops:
            return []
        if not class_names:
            raise ValueError('class_names must be a non-empty list')

        # Fire all chunks in parallel (asyncio.gather) so upstream
        # concurrency isn't artificially capped by a sequential
        # `for await`. The TokenBucket (requests_per_second) is the
        # actual global rate limit.
        chunks = [
            crops[i : i + self.max_images_per_call]
            for i in range(0, len(crops), self.max_images_per_call)
        ]
        chunk_results_list = await asyncio.gather(
            *[self._label_chunk(c, class_names) for c in chunks],
            return_exceptions=False,
        )
        results: list[VlmClassPrediction] = []
        for cr in chunk_results_list:
            results.extend(cr)
        return results

    async def _label_chunk(
        self,
        chunk: list[ItemCrop],
        class_names: list[str],
    ) -> list[VlmClassPrediction]:
        user_text = self._pack.class_user_template.format(class_names_csv=', '.join(class_names))
        user_text = f'{user_text}\n{_RESULTS_ENVELOPE}'
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
                {'role': 'system', 'content': self._pack.class_system},
                {'role': 'user', 'content': user_content},
            ],
            'temperature': 0.0,
            'max_tokens': 512,
            # Same grammar constraint as the combined call: without it a
            # server-side reasoning parser can route the whole answer to
            # the reasoning channel and leave ``content`` empty.
            **self._json_mode_kwargs(),
        }

        try:
            response = await self._post_chat(payload)
        except Exception as exc:
            logger.error(
                'vlm_labeler.label_chunk_failed',
                chunk_size=len(chunk),
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return _request_failed(chunk)

        return self._parse_class_reply(response, chunk)

    def _parse_class_reply(
        self, response: dict[str, Any], chunk: list[ItemCrop]
    ) -> list[VlmClassPrediction]:
        """Parse a class call's reply, falling back to the reasoning channel.

        Only when ``content`` yields nothing usable for any crop is the
        reasoning text tried -- a parsed ``content`` always wins.
        """
        content = _strip_markdown_fences(extract_message_content(response))
        preds = self._parse_item_response(
            content, chunk, self._fields, denylist=self._pack.proposal_denylist
        )
        if any(p.failure is None for p in preds):
            return preds
        reasoning = extract_reasoning_content(response)
        if not reasoning:
            return preds
        from_reasoning = self._parse_item_response(
            reasoning,
            chunk,
            self._fields,
            log_failures=False,
            denylist=self._pack.proposal_denylist,
        )
        if any(p.failure is None for p in from_reasoning):
            logger.info(
                'vlm_labeler.class_reply_from_reasoning',
                chunk_size=len(chunk),
                content_preview=content[:80],
            )
            return from_reasoning
        return preds

    @staticmethod
    def _parse_item_response(
        raw: str,
        chunk: list[ItemCrop],
        fields: RegionFields,
        *,
        log_failures: bool = True,
        denylist: list[str] | tuple[str, ...] = (),
    ) -> list[VlmClassPrediction]:
        """Parse the VLM's JSON-array response into one prediction per chunk crop.

        A crop with no usable entry comes back with ``class_name=''`` and
        ``failure='unparseable'`` -- distinct from a parsed entry whose
        class is empty (``failure=None``).
        """

        # Even on parse failure we preserve the raw response so the
        # curator can review what the VLM actually said.
        raw_excerpt = raw[:200]
        fallback = [
            VlmClassPrediction(
                img_id=c.img_id,
                class_name='',
                confidence='low',
                raw_response=raw_excerpt,
                failure='unparseable',
            )
            for c in chunk
        ]

        if not raw:
            if log_failures:
                logger.warning('vlm_labeler.parse_empty_response', chunk_size=len(chunk))
            return fallback

        parsed = _class_reply_entries(raw)
        if parsed is None:
            if log_failures:
                logger.warning(
                    'vlm_labeler.parse_failed',
                    raw_preview=raw[:200],
                    chunk_size=len(chunk),
                )
            return fallback

        # Map img-index → record. The VLM is told to use 1-based ``img`` ids;
        # entries with no index at all are taken positionally when their
        # count matches the chunk.
        entries = [e for e in parsed if isinstance(e, dict)]
        by_index: dict[int, dict[str, Any]] = {}
        if entries and all(e.get('img') is None for e in entries) and len(entries) == len(chunk):
            by_index = dict(enumerate(entries, start=1))
        for indexed in entries:
            raw_idx = indexed.get('img')
            if raw_idx is None:
                continue
            try:
                idx = int(raw_idx)
            except (TypeError, ValueError):
                continue
            by_index[idx] = indexed

        out: list[VlmClassPrediction] = []
        for i, crop in enumerate(chunk, start=1):
            entry = by_index.get(i)
            if entry is None:
                out.append(
                    VlmClassPrediction(
                        img_id=crop.img_id,
                        class_name='',
                        confidence='low',
                        raw_response=raw_excerpt,
                        failure='unparseable',
                    )
                )
                continue
            class_name = str(entry.get('class', '') or '').strip()
            confidence = _normalize_confidence(entry.get('confidence'))
            proposed = str(entry.get('proposed_class', '') or '').strip().lower()
            # Sanitize the proposed slug — keep [a-z0-9_], cap length 32.
            proposed = ''.join(c for c in proposed if c.isalnum() or c == '_')[:32]
            if proposed and proposal_denied(proposed, denylist):
                logger.debug('vlm_labeler.proposal_denied', proposed=proposed)
                proposed = ''
            # Per-crop raw_response: prefer the entry's `class` (the
            # VLM's actual answer for this crop) so unmatched
            # classifications land in the raw-label field instead of
            # "". Falls back to the full response excerpt when class is
            # empty.
            per_crop_raw = class_name or proposed or raw_excerpt
            make = str(entry.get('make', '') or '').strip()[:48]
            model = str(entry.get('model', '') or '').strip()[:48]
            visible_raw = entry.get(fields.visible)
            visible = bool(visible_raw) if isinstance(visible_raw, bool) else None
            out.append(
                VlmClassPrediction(
                    img_id=crop.img_id,
                    class_name=class_name,
                    confidence=confidence,
                    proposed_class=proposed,
                    raw_response=per_crop_raw,
                    make=make,
                    model=model,
                    region_visible=visible,
                )
            )
        return out

    async def label_or_propose_batch(
        self,
        crops: list[ItemCrop],
        class_names: list[str],
        *,
        images_per_call: int | None = None,
        class_catalog: str | None = None,
        cluster_hint: str | None = None,
    ) -> list[VlmClassPrediction]:
        """Label crops, but allow the VLM to propose new classes when nothing fits.

        Identical contract to :py:meth:`label_item_batch` except the
        open-vocabulary prompt is used (the VLM may answer ``__new__``
        with a ``proposed_class`` slug). Predictions for unrecognized
        items come back with ``class_name='__new__'`` and a populated
        ``proposed_class`` — callers (e.g. an auto-label pipeline) route
        those into the curator queue rather than committing them as
        labels.

        Optional ``class_catalog`` (formatted via :func:`format_class_catalog`)
        replaces the bare CSV in the prompt with grouped + described classes,
        sharply improving accuracy on ambiguous slugs. ``cluster_hint`` adds a
        per-batch bias line — pass it when labeling a single cluster's members
        to nudge the VLM toward the dominant class hypothesis. When the pack sets
        ``detector_hint_min_confidence_pct``, crops carrying a detector class get a
        per-image hint line; replies outside it are still accepted.
        """

        if not crops:
            return []
        if not class_names and not class_catalog:
            raise ValueError('class_names or class_catalog must be supplied')

        # The open-vocab prompt is verbose (per-image schema includes
        # ``proposed_class``). Smaller VLMs can produce empty responses
        # when the chunk is too dense, so default to a tighter chunk
        # size than the closed-vocab path.
        if images_per_call is not None:
            per_call = images_per_call
        else:
            per_call = self.open_images_per_call
        per_call = max(1, min(per_call, self.max_images_per_call))
        chunks = [crops[i : i + per_call] for i in range(0, len(crops), per_call)]
        chunk_results_list = await asyncio.gather(
            *[
                self._label_chunk_open(
                    c,
                    class_names,
                    class_catalog=class_catalog,
                    cluster_hint=cluster_hint,
                )
                for c in chunks
            ],
            return_exceptions=False,
        )
        results: list[VlmClassPrediction] = []
        for cr in chunk_results_list:
            results.extend(cr)
        return results

    async def _label_chunk_open(
        self,
        chunk: list[ItemCrop],
        class_names: list[str],
        *,
        class_catalog: str | None = None,
        cluster_hint: str | None = None,
    ) -> list[VlmClassPrediction]:
        if class_catalog:
            user_text = (
                'Class catalog (grouped, with brief descriptions where slugs are not '
                'self-evident):\n'
                f'{class_catalog}\n\n'
                'Label each numbered crop. Respond as one JSON object:\n'
                '{"results": [{"img": 1, "class": "<name|__new__>", '
                '"confidence": "high|medium|low", "proposed_class": "<slug or empty>"}, ...]}'
            )
        else:
            user_text = self._pack.open_class_user_template.format(
                class_names_csv=', '.join(class_names)
            )
            user_text = f'{user_text}\n{_RESULTS_ENVELOPE}'
        if self._pack.detector_hint_min_confidence_pct > 0:
            hints = [
                f'image {i}: {c.detector_class} (detector confidence {c.detector_confidence:.2f})'
                for i, c in enumerate(chunk, start=1)
                if c.detector_class and c.detector_confidence is not None
            ]
            if hints:
                user_text = (
                    f'Detector hints, one per image where the detector saw something: '
                    f'{"; ".join(hints)}. A hint, not a constraint: the detector can be wrong, '
                    'so answer from what the crop shows.\n\n'
                    f'{user_text}'
                )
        if cluster_hint:
            user_text = (
                f'Hint: these crops were grouped together by visual similarity; the '
                f'cluster\'s current dominant class hypothesis is "{cluster_hint}". '
                'Use this as a prior, not a constraint — overrule it if the crop '
                'clearly belongs to a different class.\n\n'
                f'{user_text}'
            )
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
                {'role': 'system', 'content': self._pack.open_class_system},
                {'role': 'user', 'content': user_content},
            ],
            'temperature': 0.0,
            'max_tokens': 768,
            **self._json_mode_kwargs(),
        }
        try:
            response = await self._post_chat(payload)
        except Exception as exc:
            logger.error(
                'vlm_labeler.label_chunk_open_failed',
                chunk_size=len(chunk),
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return _request_failed(chunk)
        return self._parse_class_reply(response, chunk)
