"""Judge protocol: the sheet request, the prompt, the strict verdict schema and the privacy rule.

A judge (a hosted model reached through the sheet-file exchange, or a local VLM) sees one contact
sheet per call and answers in JSON. Prompts show no labels, cluster ids or roles. Verdicts name
cells; ``odd_items`` maps them back to item and box ids through the sheet manifest.
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.services.labeling.contact_sheet import cell_to_item


if TYPE_CHECKING:
    from collections.abc import Mapping

PROMPT_VERSION = 'p1'
HOSTED_FORBIDDEN_KINDS = frozenset({'face', 'identity_text', 'plate'})
CONFIDENCE = ('high', 'medium', 'low')
ANSWERS = ('yes', 'no', 'unsure')
_PX_PER_TOKEN = 750
_FALLBACK_TOKENS = 1200


class VerdictError(ValueError):
    """The reply is not a valid verdict for the sheet."""


class PrivacyRefusalError(RuntimeError):
    """A sheet must not leave the machine, or the endpoint is not local."""


@dataclass(frozen=True)
class JudgeRequest:
    manifest: Mapping[str, Any]
    png: bytes
    prompt: str


@dataclass(frozen=True)
class RawReply:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None
    request_id: str | None = None


def assert_may_send(kind: str, *, hosted: bool) -> None:
    """Refuse before any call: face, identity-text and plate sheets stay with local judges."""
    if hosted and kind in HOSTED_FORBIDDEN_KINDS:
        raise PrivacyRefusalError(f'a {kind} sheet is never sent to a hosted model')


def estimate_tokens(png: bytes) -> int:
    """Image tokens by the vision-docs rule, width x height / 750, from the PNG header."""
    if png[:8] == b'\x89PNG\r\n\x1a\n' and len(png) >= 24:
        width, height = struct.unpack('>II', png[16:24])
        return int(max(1, width * height // _PX_PER_TOKEN))
    return _FALLBACK_TOKENS


def render_prompt(manifest: Mapping[str, Any], *, level: str) -> str:
    n = len(manifest['cells'])
    layout = manifest['layout']
    reply = 'Reply with only this JSON object and nothing else: '
    if layout == 'pair':
        rows = n // 2
        return (
            f'This image has {rows} rows. Each row shows two crops, labelled A1 and B1 in row 1, A2 and B2 in row 2, and so on. '
            f'For each row decide whether both crops show the same {level}. Judge only what is '
            f'visible; answer "unsure" when you cannot tell. {reply}'
            '{"rows": [{"row": 1, "answer": "yes" | "no" | "unsure"}, ...]} '
            f'with one entry for each row 1 to {rows}.'
        )
    if layout == 'search':
        return (
            f'This image is a contact sheet of {n} numbered cells. The cell labelled Q is the '
            f'query. Decide which of the other numbered cells show the same {level} as Q. '
            f'{reply}{{"matching_cells": [cell numbers]}} (an empty list if none match).'
        )
    cols, rows = manifest['grid']
    return (
        f'This image is a contact sheet of {n} numbered cells ({cols} x {rows}). Decide whether '
        f'every cell shows the same {level}. {reply}'
        '{"same": true | false, "odd_cells": [numbers of the cells that do not match the '
        'majority], "name": "short name of what the majority shows", '
        '"confidence": "high" | "medium" | "low"}. If every cell matches, odd_cells is [].'
    )


def _load_json(text: str) -> Any:
    body = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip())
    start, end = body.find('{'), body.rfind('}')
    if start < 0 or end < start:
        raise VerdictError('reply holds no JSON object')
    try:
        return json.loads(body[start : end + 1])
    except json.JSONDecodeError as exc:
        raise VerdictError(f'reply is not valid JSON: {exc.msg}') from exc


def _cells(value: Any, valid: set[int], field: str) -> list[int]:
    if not isinstance(value, list) or not all(isinstance(c, int) and c in valid for c in value):
        raise VerdictError(f'{field} must be a list of cell numbers on the sheet')
    return sorted(set(value))


def _cluster_verdict(doc: Mapping[str, Any], valid: set[int]) -> dict[str, Any]:
    if not isinstance(doc.get('same'), bool):
        raise VerdictError('same must be true or false')
    odd = _cells(doc.get('odd_cells', []), valid, 'odd_cells')
    if doc['same'] == bool(odd):
        raise VerdictError('same and odd_cells are not consistent')
    confidence = doc.get('confidence', 'medium')
    if confidence not in CONFIDENCE:
        raise VerdictError('confidence must be high, medium or low')
    return {
        'same': doc['same'],
        'odd_cells': odd,
        'name': str(doc.get('name', '')).strip(),
        'confidence': confidence,
    }


def _pair_verdict(doc: Mapping[str, Any], n_rows: int) -> dict[str, Any]:
    rows = doc.get('rows')
    if not isinstance(rows, list) or not all(
        isinstance(r, dict) and r.get('answer') in ANSWERS and r.get('row') in range(1, n_rows + 1)
        for r in rows
    ):
        rows = None
    if rows is None or sorted(r['row'] for r in rows) != list(range(1, n_rows + 1)):
        raise VerdictError(f'rows must answer each row 1 to {n_rows} once with yes, no or unsure')
    return {'rows': sorted(({'row': r['row'], 'answer': r['answer']} for r in rows), key=_row)}


def _row(r: Mapping[str, Any]) -> int:
    return int(r['row'])


def parse_verdict(manifest: Mapping[str, Any], text: str) -> dict[str, Any]:
    doc = _load_json(text)
    layout = manifest['layout']
    if layout == 'pair':
        return _pair_verdict(doc, len(manifest['cells']) // 2)
    valid = {c['cell'] for c in manifest['cells']}
    if layout == 'search':
        return {'matching_cells': _cells(doc.get('matching_cells'), valid, 'matching_cells')}
    return _cluster_verdict(doc, valid)


def odd_items(manifest: Mapping[str, Any], verdict: Mapping[str, Any]) -> list[tuple[str, str]]:
    """The flagged cells of a cluster verdict as ``(item_id, box_id)``."""
    return [cell_to_item(manifest, c) for c in verdict['odd_cells']]
