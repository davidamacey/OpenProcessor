"""Request parsing for the project guard: URL segments, JSON and NDJSON
bodies, bulk action lines, and the index names a body refers to.

Split out of ``guard.py``. Nothing here decides allow or refuse beyond
refusing a body it cannot parse.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import unquote

from src.services.projects.guard_errors import _refuse


_INDEX_REF_KEYS = frozenset({'index', '_index', 'indices'})


# msearch header keys (a header may override the URL's index with either
# ``index`` or ``indices``; both are read, anything else is refused).
_MSEARCH_HEADER_KEYS = frozenset(
    {'index', 'indices', 'preference', 'routing', 'request_cache', 'search_type'}
)


def _segments(url: str) -> list[str]:
    return [unquote(part) for part in url.split('?', 1)[0].split('/') if part]


def _json_value(body: Any) -> Any:
    """``body`` as parsed JSON: dicts/lists as-is (opensearch-py passes
    ``mget``/``search`` bodies to the transport unserialized), str/bytes
    parsed. Unparseable text is refused rather than skipped."""
    if body is None or isinstance(body, (dict, list)):
        return body
    text = body.decode('utf-8', 'replace') if isinstance(body, bytes) else str(body)
    if not text.strip():
        return None
    try:
        return json.loads(text)
    except ValueError as exc:
        raise _refuse('request body is not JSON') from exc


def _ndjson_lines(body: Any) -> list[Any]:
    if body is None:
        return []
    if isinstance(body, list):
        return body
    text = body.decode('utf-8', 'replace') if isinstance(body, bytes) else str(body)
    lines = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            lines.append(json.loads(line))
        except ValueError as exc:
            raise _refuse('multi-doc body line is not JSON') from exc
    return lines


_BULK_DOCUMENT_OPS = ('index', 'create', 'update')


def _bulk_lines(body: Any) -> list[Any]:
    """The lines of a bulk body, action lines parsed and document lines left alone.

    Only an action line can name an index, and a document line of an ingest request
    carries a 1,024-float vector, so parsing every line parsed some 25 KB of floats per
    item for nothing. A document line keeps its slot as ``None`` and must still look
    like a JSON object; OpenSearch rejects one that is not valid JSON.
    """
    if body is None or isinstance(body, list):
        return _ndjson_lines(body)
    text = body.decode('utf-8', 'replace') if isinstance(body, bytes) else str(body)
    lines: list[Any] = []
    expect_document = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if expect_document:
            if not (stripped.startswith('{') and stripped.endswith('}')):
                raise _refuse('multi-doc body line is not JSON')
            lines.append(None)
            expect_document = False
            continue
        try:
            action = json.loads(stripped)
        except ValueError as exc:
            raise _refuse('multi-doc body line is not JSON') from exc
        lines.append(action)
        if isinstance(action, dict) and len(action) == 1:
            expect_document = next(iter(action)) in _BULK_DOCUMENT_OPS
    return lines


def _index_refs(value: Any) -> list[str]:
    """Every string under an ``index``/``_index``/``indices`` key anywhere
    in a query body (terms lookups, ``more_like_this`` docs, ``mget``
    docs, ...)."""
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in _INDEX_REF_KEYS:
                if isinstance(item, str):
                    found.extend(item.split(','))
                elif isinstance(item, list):
                    found.extend(x for x in item if isinstance(x, str))
            found.extend(_index_refs(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_index_refs(item))
    return found


def _body_targets(action: str, body: Any, *, require_each: bool = False) -> list[str]:
    """Indexes a request body names. ``require_each``: the URL names no
    index, so every line / doc must name its own."""
    targets: list[str] = []
    if action == '_bulk':
        lines = _bulk_lines(body)
        i = 0
        while i < len(lines):
            line = lines[i]
            op = next(iter(line), None) if isinstance(line, dict) and len(line) == 1 else None
            if op not in ('index', 'create', 'update', 'delete') or not isinstance(line[op], dict):
                raise _refuse('a bulk action line is malformed')
            name = line[op].get('_index')
            if name:
                targets.append(name)
            elif require_each:
                raise _refuse('a bulk action names no index')
            i += 1 if op == 'delete' else 2
        return targets
    if action == '_msearch':
        lines = _ndjson_lines(body)
        for header in lines[0::2]:
            if not isinstance(header, dict):
                raise _refuse('msearch header is not an object')
            unknown = set(header) - _MSEARCH_HEADER_KEYS
            if unknown:
                raise _refuse(f'msearch header key(s) {sorted(unknown)} are not allowed')
            named = [
                part
                for key in ('index', 'indices')
                for value in [header.get(key)]
                if value
                for part in (value.split(',') if isinstance(value, str) else value)
            ]
            if named:
                targets.extend(named)
            elif require_each:
                raise _refuse('an msearch header names no index')
        for query in lines[1::2]:
            targets.extend(_index_refs(query))
        return targets
    payload = _json_value(body)
    if action == '_mget' and require_each:
        docs = (payload or {}).get('docs') if isinstance(payload, dict) else None
        if not docs or any(not isinstance(d, dict) or not d.get('_index') for d in docs):
            raise _refuse('an mget doc names no index')
    return targets + _index_refs(payload)


def _handle_ids(segments: list[str], body: Any, params: Any) -> list[str]:
    ids: list[str] = []
    if len(segments) > 2:
        ids.extend(segments[2].split(','))
    payload = _json_value(body)
    raw = (payload or {}).get('scroll_id') if isinstance(payload, dict) else None
    if raw is None and isinstance(params, dict):
        raw = params.get('scroll_id')
    if isinstance(raw, bytes):
        raw = raw.decode('utf-8', 'replace')
    if isinstance(raw, str):
        ids.extend(raw.split(','))
    elif isinstance(raw, list):
        ids.extend(str(x) for x in raw)
    return ids
