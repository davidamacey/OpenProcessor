"""The assist and attributes calls of the VLM labeler.

``assist_batch`` classifies item crops and reports the pack's item attributes in
one call; ``attributes_batch`` reports attributes alone (items without classify,
and box sub-crops). Both are suggestions: the caller decides what is written.
Class parsing is shared with :mod:`~src.services.labeling.vlm_labeler_classify`.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import httpx

from src.core.logging import get_logger
from src.services.labeling.vlm_attributes import (
    AppliesTo,
    AttributeSpec,
    parse_attributes,
    render_attribute_block,
    specs_for,
)
from src.services.labeling.vlm_client import extract_message_content, extract_reasoning_content
from src.services.labeling.vlm_labeler_classify import ClassifyOps
from src.services.labeling.vlm_labeler_core import _RESULTS_ENVELOPE, _b64_jpeg
from src.services.labeling.vlm_models import ItemCrop, VlmAttributesPrediction, VlmClassPrediction
from src.services.labeling.vlm_reply_parse import (
    _class_reply_entries,
    _request_failed,
    _strip_markdown_fences,
    entries_by_index,
)


if TYPE_CHECKING:
    from collections.abc import Sequence


logger = get_logger(__name__)


class PackAttributesError(ValueError):
    """The pack cannot serve the call: it declares no attributes of the kind
    asked for, or its prompt for the call is blank."""


def _user_content(text: str, chunk: list[ItemCrop]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{'type': 'text', 'text': text}]
    content.extend(
        {
            'type': 'image_url',
            'image_url': {'url': f'data:image/jpeg;base64,{_b64_jpeg(crop.jpeg_bytes)}'},
        }
        for crop in chunk
    )
    return content


class AssistOps(ClassifyOps):
    """Classify-with-attributes and attributes-only calls."""

    def _chunks(self, crops: list[ItemCrop], per_call: int | None) -> list[list[ItemCrop]]:
        size = self.open_images_per_call if per_call is None else per_call
        size = max(1, min(size, self.max_images_per_call))
        return [crops[i : i + size] for i in range(0, len(crops), size)]

    def _declared(self, applies_to: AppliesTo, *prompts: str) -> list[AttributeSpec]:
        specs = specs_for(self._pack.attributes, applies_to)
        if not specs:
            msg = f'pack {self._pack.name!r} declares no {applies_to} attributes'
            raise PackAttributesError(msg)
        for name in prompts:
            if not getattr(self._pack, name).strip():
                msg = f'pack {self._pack.name!r} has no {name}'
                raise PackAttributesError(msg)
        return specs

    async def assist_batch(
        self,
        crops: list[ItemCrop],
        class_names: list[str],
        *,
        class_catalog: str | None = None,
        images_per_call: int | None = None,
    ) -> list[VlmClassPrediction]:
        """Classify ``crops`` (the VLM may answer ``__new__`` with a proposed
        slug) and report the pack's item attributes. One prediction per crop,
        in order; ``attributes`` holds declared keys with allowed values only.
        Raises :class:`PackAttributesError` when the pack cannot serve it."""
        specs = self._declared('item', 'assist_system', 'assist_user_template')
        if not crops:
            return []
        if not class_names and not class_catalog:
            raise ValueError('class_names or class_catalog must be supplied')
        results = await asyncio.gather(
            *[
                self._assist_chunk(c, class_names, class_catalog, specs)
                for c in self._chunks(crops, images_per_call)
            ]
        )
        return [p for chunk in results for p in chunk]

    async def _assist_chunk(
        self,
        chunk: list[ItemCrop],
        class_names: list[str],
        class_catalog: str | None,
        specs: list[AttributeSpec],
    ) -> list[VlmClassPrediction]:
        text = self._pack.assist_user_template.format(
            class_list=class_catalog or ', '.join(class_names),
            attribute_block=render_attribute_block(specs),
            n_images=len(chunk),
        )
        payload = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': self._pack.assist_system},
                {'role': 'user', 'content': _user_content(f'{text}\n{_RESULTS_ENVELOPE}', chunk)},
            ],
            'temperature': 0.0,
            'max_tokens': 1024,
            **self._json_mode_kwargs(),
        }
        try:
            response = await self._post_chat(payload)
        except (httpx.HTTPError, ValueError) as exc:
            logger.error(
                'vlm_labeler.assist_chunk_failed',
                chunk_size=len(chunk),
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return _request_failed(chunk)
        return self._parse_class_reply(response, chunk, specs)

    async def attributes_batch(
        self,
        crops: list[ItemCrop],
        *,
        applies_to: AppliesTo = 'item',
        images_per_call: int | None = None,
    ) -> list[VlmAttributesPrediction]:
        """Report the pack's ``applies_to`` attributes for each crop, no
        classification. One prediction per crop, in order."""
        specs = self._declared(applies_to, 'attributes_system', 'attributes_user_template')
        if not crops:
            return []
        results = await asyncio.gather(
            *[self._attributes_chunk(c, specs) for c in self._chunks(crops, images_per_call)]
        )
        return [p for chunk in results for p in chunk]

    async def _attributes_chunk(
        self, chunk: list[ItemCrop], specs: list[AttributeSpec]
    ) -> list[VlmAttributesPrediction]:
        text = self._pack.attributes_user_template.format(
            attribute_block=render_attribute_block(specs), n_images=len(chunk)
        )
        payload = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': self._pack.attributes_system},
                {'role': 'user', 'content': _user_content(f'{text}\n{_RESULTS_ENVELOPE}', chunk)},
            ],
            'temperature': 0.0,
            'max_tokens': 768,
            **self._json_mode_kwargs(),
        }
        try:
            response = await self._post_chat(payload)
        except (httpx.HTTPError, ValueError) as exc:
            logger.error(
                'vlm_labeler.attributes_chunk_failed',
                chunk_size=len(chunk),
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return [
                VlmAttributesPrediction(img_id=c.img_id, failure='request_failed') for c in chunk
            ]
        content = _strip_markdown_fences(extract_message_content(response))
        preds = _parse_attributes_reply(content, chunk, specs)
        if any(p.failure is None for p in preds):
            return preds
        reasoning = extract_reasoning_content(response)
        if reasoning:
            from_reasoning = _parse_attributes_reply(reasoning, chunk, specs)
            if any(p.failure is None for p in from_reasoning):
                return from_reasoning
        return preds


def _parse_attributes_reply(
    raw: str, chunk: list[ItemCrop], specs: Sequence[AttributeSpec]
) -> list[VlmAttributesPrediction]:
    entries = _class_reply_entries(raw) if raw else None
    by_index = entries_by_index(entries, len(chunk)) if entries is not None else {}
    out: list[VlmAttributesPrediction] = []
    for i, crop in enumerate(chunk, start=1):
        entry = by_index.get(i)
        if entry is None:
            out.append(VlmAttributesPrediction(img_id=crop.img_id, failure='unparseable'))
            continue
        attributes, invalid = parse_attributes(entry.get('attributes'), specs)
        out.append(
            VlmAttributesPrediction(
                img_id=crop.img_id, attributes=attributes, invalid_values=invalid
            )
        )
    return out


__all__ = ['AssistOps', 'PackAttributesError']
