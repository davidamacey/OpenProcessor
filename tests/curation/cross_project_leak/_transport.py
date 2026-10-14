"""A fake OpenSearch transport that records every access with its own parser."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote

import httpx
from opensearchpy.exceptions import NotFoundError

from curation.query_fakes import _aggregate, matches


if TYPE_CHECKING:
    from pathlib import Path


def _body_json(body: Any) -> Any:
    if body is None:
        return None
    if isinstance(body, (dict, list)):
        return body
    text = body.decode('utf-8', 'replace') if isinstance(body, bytes) else str(body)
    try:
        return json.loads(text)
    except ValueError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def _body_lines(body: Any) -> list[Any]:
    if body is None:
        return []
    if isinstance(body, list):
        return body
    text = body.decode('utf-8', 'replace') if isinstance(body, bytes) else str(body)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def touched_indexes(url: str, body: Any) -> set[str]:
    """Every index a request reaches, independent of the guard's parser.
    A search-shaped request with no index at all is recorded as ``*``."""
    parts = [unquote(p) for p in url.split('?', maxsplit=1)[0].split('/') if p]
    out: set[str] = set()
    if parts and not parts[0].startswith('_'):
        out |= set(parts[0].split(','))
    if parts[:3] == ['_plugins', '_knn', 'warmup'] and len(parts) > 3:
        out |= set(parts[3].split(','))
    action = next((p for p in parts if p.startswith('_')), '')
    url_index = bool(out)
    if action == '_mget':
        for doc in (_body_json(body) or {}).get('docs', []):
            out.add(doc.get('_index') or ('' if url_index else '*'))
    elif action == '_bulk':
        for line in _body_lines(body):
            for op in ('index', 'create', 'update', 'delete'):
                if isinstance(line.get(op), dict):
                    out.add(line[op].get('_index') or ('' if url_index else '*'))
    elif action == '_msearch':
        for header in _body_lines(body)[0::2]:
            named = False
            for key in ('index', 'indices'):
                target = header.get(key)
                if target:
                    named = True
                    out |= set(target.split(',') if isinstance(target, str) else target)
            if not named and not url_index:
                out.add('*')
    elif action in ('_search', '_count') and not url_index and parts[:2] != ['_search', 'scroll']:
        out.add('*')
    out.discard('')
    return out


_FAKE_VLM_HOST = 'vlm.leak-test'


def _fake_vlm_reply(request: httpx.Request) -> httpx.Response:
    """An OpenAI-shaped chat completion answering every image in the
    request with one confident verdict: the first catalog class, region
    visible and verified, a short text read."""
    payload = json.loads(request.content or b'{}')
    n_images = sum(
        1
        for message in payload.get('messages', [])
        if isinstance(message.get('content'), list)
        for part in message['content']
        if isinstance(part, dict) and part.get('type') == 'image_url'
    )
    answers = [
        {
            'img': i + 1,
            'class_id': 0,
            'confidence': 'high',
            'region_visible': True,
            'is_region': True,
            'is_plate': True,
            'region_text': 'AB12',
            'text': 'AB12',
            'reason': 'clear',
        }
        for i in range(max(n_images, 1))
    ]
    body = {
        'id': 'leak-test',
        'object': 'chat.completion',
        'model': payload.get('model', 'fake'),
        'choices': [
            {
                'index': 0,
                'message': {'role': 'assistant', 'content': json.dumps(answers)},
                'finish_reason': 'stop',
            }
        ],
        'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
    }
    return httpx.Response(200, json=body, request=request)


def _query_matches(doc_id: str, doc: dict[str, Any], query: Any) -> bool:
    """Evaluate the query DSL subset ``tests/curation/query_fakes.py``
    understands (plus ``ids``); an unsupported clause matches everything, so
    a route that reaches an index still sees (and would leak) its docs."""
    if not query:
        return True
    if 'ids' in query:
        return doc_id in (query['ids'].get('values') or [])
    try:
        return matches({**doc, '_id': doc_id, 'crop_id': doc.get('crop_id', doc_id)}, query)
    except (NotImplementedError, KeyError, TypeError, ValueError):
        return True


class _FakeTransport:
    """The bottom of the fake: answers OpenSearch REST calls from a per-
    index doc store, and applies writes to it. Queries, sizes and
    aggregations are evaluated (``tests/curation/query_fakes.py``) so each
    write route finds the state it acts on; an unsupported clause matches
    every doc of the index it names, so a reached index still shows up."""

    def __init__(self, store: dict[str, dict[str, dict[str, Any]]]) -> None:
        from opensearchpy.serializer import JSONSerializer

        self.serializer = JSONSerializer()
        self.store = store
        self.writes: list[str] = []  # index names written
        self.received: list[set[str]] = []  # indexes of every request that got past the guard
        self._scrolls = 0
        self._tasks: dict[str, Any] = {}

    def _hits(self, indices: list[str], query: Any = None) -> list[dict[str, Any]]:
        return [
            {'_index': idx, '_id': doc_id, '_source': doc, '_seq_no': 1, '_primary_term': 1}
            for idx in indices
            for doc_id, doc in self.store.get(idx, {}).items()
            if _query_matches(doc_id, doc, query)
        ]

    def _search(self, indices: list[str], body: Any, params: Any) -> dict[str, Any]:
        spec = _body_json(body) or {}
        spec = spec if isinstance(spec, dict) else {}
        hits = self._hits(indices, spec.get('query'))
        aggs: dict[str, Any] = {}
        if spec.get('aggs') or spec.get('aggregations'):
            try:
                aggs = _aggregate(
                    [h['_source'] for h in hits], spec.get('aggs') or spec['aggregations']
                )
            except (NotImplementedError, KeyError, TypeError, ValueError):
                aggs = {}
        size = spec.get('size', 10)
        resp: dict[str, Any] = {
            'hits': {
                'total': {'value': len(hits), 'relation': 'eq'},
                'hits': hits[: size if isinstance(size, int) else 10],
            },
            'aggregations': aggs,
        }
        if (params or {}).get('scroll'):
            self._scrolls += 1
            resp['_scroll_id'] = f'scroll-{self._scrolls}'
        return resp

    def _write(self, index: str, doc_id: str, doc: dict[str, Any] | None, *, merge: bool) -> None:
        self.writes.append(index)
        docs = self.store.setdefault(index, {})
        if doc is None:
            docs.pop(doc_id, None)
        elif merge:
            docs.setdefault(doc_id, {}).update(doc)
        else:
            docs[doc_id] = doc

    async def perform_request(  # noqa: PLR0911 - one branch per endpoint shape
        self,
        method: str,
        url: str,
        params: Any = None,
        body: Any = None,
        **_kwargs: Any,
    ) -> Any:
        self.received.append(touched_indexes(url, body))
        parts = [unquote(p) for p in url.split('?', maxsplit=1)[0].split('/') if p]
        indices = parts[0].split(',') if parts and not parts[0].startswith('_') else []
        action = next((p for p in parts[1:] if p.startswith('_')), parts[0] if parts else '')
        if method == 'HEAD':
            return True
        if parts[:2] == ['_search', 'scroll']:
            if method == 'DELETE':
                return {'succeeded': True}
            return {'_scroll_id': (_body_json(body) or {}).get('scroll_id'), 'hits': {'hits': []}}
        if action == '_doc' and method == 'GET':
            doc_id = parts[2] if len(parts) > 2 else ''
            doc = self.store.get(indices[0], {}).get(doc_id)
            if doc is None:
                raise NotFoundError(404, 'not_found', {'found': False})
            return {
                '_index': indices[0],
                '_id': doc_id,
                'found': True,
                '_source': doc,
                '_seq_no': 1,
                '_primary_term': 1,
            }
        if action == '_search':
            return self._search(indices, body, params)
        if action == '_msearch':
            lines = _body_lines(body)
            responses = []
            for header, query in zip(lines[0::2], lines[1::2], strict=False):
                named = header.get('index') or header.get('indices') or indices
                targets = named.split(',') if isinstance(named, str) else list(named)
                responses.append(self._search(targets, query, None))
            return {'responses': responses}
        if action == '_count':
            spec = _body_json(body) or {}
            query = spec.get('query') if isinstance(spec, dict) else None
            return {'count': len(self._hits(indices, query))}
        if action == '_mget':
            return {'docs': self._mget(indices, body)}
        if action == '_bulk':
            return self._bulk(indices, body)
        if action in ('_update_by_query', '_delete_by_query'):
            for idx in indices:
                self.writes.append(idx)
            result = {'updated': 0, 'deleted': 0, 'total': 0, 'failures': []}
            wait = (params or {}).get('wait_for_completion', b'true')
            if (wait.decode() if isinstance(wait, bytes) else str(wait)).lower() == 'false':
                self._scrolls += 1
                self._tasks[f'node:{self._scrolls}'] = result
                return {'task': f'node:{self._scrolls}'}
            return result
        if parts[:1] == ['_tasks']:
            return {'completed': True, 'response': self._tasks.get(parts[1], {})}
        if action == '_mapping':
            return {idx: {'mappings': {'properties': {}}} for idx in indices}
        if action in ('_doc', '_create', '_update') and method in ('PUT', 'POST', 'DELETE'):
            doc_id = parts[2] if len(parts) > 2 else f'auto-{len(self.writes)}'
            payload = _body_json(body) or {}
            if method == 'DELETE':
                self._write(indices[0], doc_id, None, merge=False)
            elif action == '_update':
                # `bump_config_revision`'s
                # painless script + upsert body needs real semantics here,
                # not the generic doc-merge below -- mirrors
                # tests/curation/_fake_config_opensearch.py's
                # FakeConfigOpenSearch.update(). A brand-new doc gets the
                # upsert body verbatim (real OpenSearch never runs the
                # script on the insert path); an existing doc gets the
                # script's increment applied.
                current = self.store.get(indices[0], {}).get(doc_id)
                script_source = (payload.get('script') or {}).get('source', '')
                if current is None and 'config_revision' in script_source:
                    self._write(indices[0], doc_id, dict(payload.get('upsert') or {}), merge=False)
                elif current is not None and 'config_revision += 1' in script_source:
                    bumped = dict(current)
                    bumped['config_revision'] = int(bumped.get('config_revision', 0)) + 1
                    self._write(indices[0], doc_id, bumped, merge=False)
                else:
                    self._write(indices[0], doc_id, payload.get('doc') or {}, merge=True)
            else:
                self._write(indices[0], doc_id, payload, merge=False)
            return {'_id': doc_id, 'result': 'updated', '_seq_no': 2, '_primary_term': 1}
        if method in ('PUT', 'DELETE') and indices and len(parts) == 1:
            return {'acknowledged': True}
        if parts[:1] == ['_cat']:
            return []
        return {'hits': {'total': {'value': 0}, 'hits': []}, 'acknowledged': True}

    def _bulk(self, indices: list[str], body: Any) -> dict[str, Any]:
        lines = _body_lines(body)
        items = []
        i = 0
        while i < len(lines):
            op, meta = next(iter(lines[i].items()))
            idx = meta.get('_index') or (indices[0] if indices else '')
            doc_id = meta.get('_id') or f'auto-{len(self.writes)}'
            if op == 'delete':
                self._write(idx, doc_id, None, merge=False)
                i += 1
            else:
                src = lines[i + 1] if i + 1 < len(lines) else {}
                if op == 'update':
                    self._write(idx, doc_id, src.get('doc') or {}, merge=True)
                else:
                    self._write(idx, doc_id, src, merge=False)
                i += 2
            items.append({op: {'_index': idx, '_id': doc_id, 'status': 200}})
        return {'errors': False, 'items': items}

    def _mget(self, indices: list[str], body: Any) -> list[dict[str, Any]]:
        spec = _body_json(body) or {}
        entries = spec.get('docs') or [{'_id': i} for i in spec.get('ids', [])]
        out = []
        for entry in entries:
            idx = entry.get('_index') or (indices[0] if indices else '')
            doc = self.store.get(idx, {}).get(entry.get('_id'))
            out.append(
                {
                    '_index': idx,
                    '_id': entry.get('_id'),
                    'found': doc is not None,
                    '_source': doc or {},
                    '_seq_no': 1,
                    '_primary_term': 1,
                }
            )
        return out

    async def close(self) -> None:
        return None


class _RecordingTransport:
    """Sits *above* the guard: records every request (bound slug, method,
    url, indexes) before the guard decides."""

    def __init__(self, inner: Any, log: list[tuple[str | None, str, str, str]]) -> None:
        self._inner = inner
        self._log = log

    async def perform_request(
        self, method: str, url: str, params: Any = None, body: Any = None, **kw: Any
    ) -> Any:
        from src.config.project_context import try_current_project

        bound = try_current_project()
        slug = bound.record.slug if bound is not None else None
        self._log.extend((slug, method, url, idx) for idx in touched_indexes(url, body))
        return await self._inner.perform_request(method, url, params=params, body=body, **kw)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


@dataclass
class LeakEnv:
    app: Any
    records: dict[str, Any]
    accesses: list[tuple[str | None, str, str, str]]
    transport: _FakeTransport
    own_dirs: dict[str, list[Path]]
    root: Path
    events: list[dict[str, Any]] = field(default_factory=list)
