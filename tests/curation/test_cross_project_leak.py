"""The isolation proof (projects_plan.md §10 P1): every scoped curation
route, swept first as one project and then as the other (alpha then beta,
and beta then alpha), reads and writes only the bound project's
OpenSearch indexes, directories, caches and event stream.

Why two passes: a process cache primed by the first project is read back
by the second one. A single sweep as one project can never see that.

Setup: three projects -- ``default`` (the env-named indexes and dirs),
``alpha`` and ``beta`` -- on a fake OpenSearch transport behind the real
project guard. Each is seeded with its own items, images, labels and
classes, and with filesystem state under its own resources (job state,
a training run, an FP centroid store). Every id and class name embeds
the project slug, so a leak is greppable.

Per route, both passes assert:
- no OpenSearch access to another project's index (recorded *above* the
  guard with this file's own parser, so a refused or swallowed access
  still counts, and ``mget``/``msearch``/``bulk`` bodies are read whatever
  their Python type);
- the bound project never drifts;
- no other project's marker in the response, and every served URL is
  under the bound project's prefix;
- no event reaching another project's stream, and no event on the
  global stream;
- no 5xx unless listed in ``EXPECTED_5XX`` with a reason;
- a mutating route sends a valid body (a 422 means "unmapped body") and
  really writes (to the bound project's indexes or dirs) unless listed in
  ``NO_WRITE`` with a reason.
Across the second pass, the first project's directories are byte-for-byte
unchanged.

Routes are enumerated from the app's route table, so a route added later
is covered automatically; a new path parameter, streaming route or
mutating route fails the test until it is mapped below.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import subprocess  # nosec B404 - only patched to refuse, never called
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from opensearchpy import AsyncOpenSearch
from opensearchpy.exceptions import NotFoundError


if TYPE_CHECKING:
    from pathlib import Path


API = '/curation'
SCOPED = f'{API}/projects/{{project}}'
SLUGS = ('default', 'alpha', 'beta')
CLASS_NAMES = {'default': 'default_cardinal', 'alpha': 'alpha_zebra', 'beta': 'beta_heron'}
EMBED_DIM = 8


def route_params(slug: str) -> dict[str, str]:
    """Every path parameter a scoped route may carry, filled with ``slug``'s
    ids. A route with a parameter missing here fails ("unmapped route")."""
    return {
        'crop_id': f'{slug}-item-0001',
        'image_id': f'{slug}-img-0001',
        'class_id': '2',
        'cluster_id': '1',
        'job_id': f'{slug}-job-0001',
        'campaign_id': f'{slug}-campaign-0001',
        'name': f'{slug}-model',
        'model_name': f'{slug}-model',
        'tab': 'uncertainty',
        'alias': f'{slug}-source',
        'artifact': 'results.csv',
    }


def route_bodies(slug: str) -> dict[tuple[str, str], dict[str, Any]]:
    """A minimal valid request for every mutating route, as ``slug``.
    ``{'json': ...}`` / ``{'files': ..., 'data': ...}`` / ``{'params': ...}``
    are passed straight to ``TestClient.request``. A mutating route missing
    here that answers 422 fails with "unmapped body"."""
    item = f'{slug}-item-0001'
    proposal = f'{slug}-item-0002'
    img = f'/data/{slug}/{slug}-img-0001.jpg'
    return {
        ('POST', '/bakeoff/run'): {'json': {'job_id': f'{slug}-job-0001'}},
        ('PUT', '/crops/{crop_id}/label'): {'json': {'class_id': 1}},
        ('PUT', '/crops/batch_label'): {'json': {'crop_ids': [proposal], 'class_id': 1}},
        ('POST', '/crops/move'): {'json': {'crop_ids': [proposal], 'cluster_id': 1}},
        ('POST', '/crops/flag_new_class'): {'json': {'crop_ids': [item], 'note': 'n'}},
        ('POST', '/crops/batch_exclude'): {'json': {'crop_ids': [proposal], 'reason': 'r'}},
        ('POST', '/crops/batch_unexclude'): {'json': {'crop_ids': [proposal]}},
        ('POST', '/classes'): {'json': {'name': f'{slug}_newclass'}},
        ('PUT', '/classes/{class_id}'): {'json': {'group': f'{slug}_group'}},
        ('POST', '/classes/merge'): {'json': {'source_id': 2, 'target_id': 1}},
        ('POST', '/crops/label/undo_batch'): {'json': {'crop_ids': [item]}},
        ('POST', '/crops/{crop_id}/discard'): {'json': {}},
        ('POST', '/crops/discard_batch'): {'json': {'crop_ids': [item]}},
        ('POST', '/crops/region/undo_batch'): {'json': {'crop_ids': [item]}},
        ('POST', '/events/publish'): {
            'json': {'type': 'crop.classified', 'crop_id': item, 'class_id': 1}
        },
        ('POST', '/export/yolo'): {'json': {'version_tag': f'{slug}-v1'}},
        ('POST', '/export/single_class'): {'json': {'version_tag': f'{slug}-v1', 'class_ids': [1]}},
        ('POST', '/vlm/label_batch'): {'json': {'crop_ids': [item]}},
        ('POST', '/vlm/verify_regions'): {'json': {'crop_ids': [item]}},
        ('POST', '/vlm/verify_region_batch'): {'json': {'items': [{'crop_id': item}]}},
        ('POST', '/vlm/region_visible_batch'): {'json': {'items': [{'crop_id': item}]}},
        ('POST', '/ingest/image'): {'json': {'path': img}},
        ('POST', '/ingest/batch'): {'json': {'items': [{'path': img}]}},
        ('POST', '/import_labels'): {
            'json': {'image_path': img, 'label_txt_path': img.replace('.jpg', '.txt')}
        },
        ('POST', '/import_labels/batch'): {
            'json': {'items': [{'image_path': img, 'label_txt_path': img.replace('.jpg', '.txt')}]}
        },
        ('POST', '/ingest/path_lookup'): {'json': {'image_paths': [img]}},
        ('POST', '/ingest/upload'): {
            'files': [('images', (f'{slug}-up.jpg', b'\xff\xd8\xff\xd9', 'image/jpeg'))],
        },
        ('POST', '/probe/run'): {'json': {'job_id': f'{slug}-job-0001'}},
        ('PUT', '/crops/{crop_id}/region'): {'json': {'region_bbox_norm': [0.1, 0.1, 0.4, 0.4]}},
        ('PATCH', '/crops/{crop_id}/region_meta'): {'json': {'region_text': f'{slug}TXT'}},
        ('PUT', '/crops/batch_region'): {
            'json': {'crop_ids': [proposal], 'region_bbox_norm': [0.1, 0.1, 0.4, 0.4]}
        },
        ('POST', '/regions/batch_status'): {
            'json': {'crop_ids': [item], 'region_status': 'false_positive'}
        },
        ('POST', '/test_holdout/freeze'): {'json': {'percent': 10}},
        ('POST', '/review/new_class_proposals/resolve'): {
            'json': {'label': f'{slug}-proposal', 'class_id': 1}
        },
        ('POST', '/scores/compute'): {'json': {}},
        ('POST', '/select/diverse'): {'json': {'k': 1}},
        ('PUT', '/settings'): {'json': {'defaults': {}}},
        ('POST', '/train/preflight'): {'json': {}},
        ('POST', '/train/start'): {'json': {}},
        ('POST', '/train/start_campaign'): {
            'json': {
                'dataset_export_dir': f'/exports/{slug}-v1',
                'runs': [{'model_size': 'n'}],
            }
        },
        ('POST', '/train/promote/{job_id}'): {'json': {'triton_name': f'{slug}_model'}},
    }


# Mutating routes that legitimately issue no write in the fixture, and why.
NO_WRITE: dict[tuple[str, str], str] = {
    ('POST', '/bakeoff/run'): 'needs a finished training run; answers 404/409 first',
    (
        'POST',
        '/classes/merge',
    ): 'the fake search returns every doc, so the frozen-holdout guard fires',
    ('POST', '/crops/{crop_id}/vlm_dismiss'): 'the seeded item carries no VLM suggestion',
    ('POST', '/classes/{class_id}/deprecate'): (
        'the fake search ignores the query, so every class looks referenced (409)'
    ),
    ('POST', '/clusters/refine/{cluster_id}'): 'one item: nothing to refine',
    ('POST', '/clusters/auto_promote'): 'no cluster reaches the promote purity bar',
    ('POST', '/crops/label/undo_batch'): 'no label history to undo',
    ('POST', '/crops/{crop_id}/region/undo'): 'no region history to undo',
    ('POST', '/crops/region/undo_batch'): 'no region history to undo',
    ('POST', '/crops/{crop_id}/vlm_dismiss/undo'): 'nothing dismissed to undo',
    ('POST', '/events/publish'): 'publishes an event (checked separately), writes no data',
    ('POST', '/vlm/label_batch'): 'the VLM is unreachable (network disabled)',
    ('POST', '/vlm/verify_regions'): 'the VLM is unreachable (network disabled)',
    ('POST', '/vlm/verify_region_batch'): 'the VLM is unreachable (network disabled)',
    ('POST', '/vlm/region_visible_batch'): 'the VLM is unreachable (network disabled)',
    ('POST', '/vlm/label_cluster/{cluster_id}'): 'the VLM is unreachable (network disabled)',
    ('POST', '/ingest/image'): 'Triton is down in the fixture: ingest fails before a write',
    ('POST', '/ingest/batch'): 'Triton is down in the fixture: ingest fails before a write',
    ('POST', '/import_labels'): 'the label file does not exist on disk',
    ('POST', '/import_labels/batch'): 'the label file does not exist on disk',
    ('POST', '/ingest/path_lookup'): 'read-only lookup under POST',
    ('POST', '/ingest/upload'): 'the upload is not a decodable image: refused before a write',
    ('DELETE', '/models/{model_name}'): 'no such promoted model: 404',
    ('POST', '/pipeline/auto_label/cancel'): 'no auto-label job running',
    ('POST', '/probe/run'): 'no finished training run to probe',
    ('POST', '/probe/cancel'): 'no probe job running',
    ('POST', '/regions/clusters/refine/{cluster_id}'): 'one region: nothing to refine',
    ('POST', '/review/new_class_proposals/resolve'): 'no pending proposal with that label',
    ('POST', '/scores/cancel'): 'no scoring job running',
    ('POST', '/select/cancel'): 'no selection job running',
    ('POST', '/viz/projection/cancel'): 'no projection job running',
    ('POST', '/train/preflight'): 'read-only validation under POST',
    ('POST', '/train/start'): 'no export to train on: refused by preflight',
    ('POST', '/train/start_campaign'): 'the export dir does not exist: refused by preflight',
    ('POST', '/train/cancel/{job_id}'): 'no such run',
    ('POST', '/train/cancel_campaign/{campaign_id}'): 'no such campaign',
    ('POST', '/train/promote/{job_id}'): 'no such run',
    ('POST', '/train/reload_promoted'): 'Triton is down in the fixture',
    ('POST', '/test_holdout/freeze'): 'one image: 10% of it freezes nothing',
    ('POST', '/viz/projection/rebuild'): 'the background job is refused by the pool floor',
    ('POST', '/cluster/umap/rebuild'): 'the background job is refused by the pool floor',
    ('POST', '/select/diverse'): 'runs as a background job; the job reads only',
}

# Routes that answer 5xx in the fixture for a reason that is not isolation.
EXPECTED_5XX: dict[tuple[str, str], str] = {
    ('POST', '/train/reload_promoted'): 'Triton is down in the fixture',
    ('POST', '/ingest/batch'): 'no PE encoder in the fixture (Triton is down)',
    ('POST', '/ingest/upload'): 'no PE encoder in the fixture (Triton is down)',
    ('POST', '/cluster/umap/rebuild'): 'UMAP spectral init needs more points than the fixture has',
}

# Known leaks owned by P2 (cutover/projects-workers, projects_plan.md §5):
# process-global state P1 did not create and P2 makes per project. Each
# entry: (method, template) -> the foreign-project evidence it may show.
# Anything else is a P1 failure.
P2_DEFERRED: dict[tuple[str, str], str] = {
    ('POST', '/pipeline/auto_label/start'): 'global auto-label trigger/state dir (P2)',
    ('POST', '/pipeline/auto_label'): 'global auto-label trigger/state dir (P2)',
    ('GET', '/pipeline/auto_label/status'): 'global auto-label state dir (P2)',
    ('GET', '/pipeline/auto_label/history'): 'global auto-label state dir (P2)',
    ('GET', '/pipeline/auto_label/history/{job_id}'): 'global auto-label state dir (P2)',
    ('GET', '/pipeline/stats'): 'global auto-label state dir (P2)',
    ('GET', '/train/runs'): 'global training staging dir (P2)',
    ('GET', '/train/status'): 'global training staging dir (P2)',
    ('POST', '/vlm/label_cluster/{cluster_id}'): 'queues the global auto-label job (P2)',
}

# Long-lived SSE streams: the per-project delivery they serve is proven by
# the event checks below and tests/projects/test_event_hub_project_filter.py.
STREAMING_ROUTES: frozenset[str] = frozenset({f'{SCOPED}/events', f'{SCOPED}/pipeline/events'})


def _docs(slug: str) -> dict[str, dict[str, dict[str, Any]]]:
    """One project's seed data, keyed by IndexRole value -> doc id -> doc."""
    item_id = f'{slug}-item-0001'
    image_id = f'{slug}-img-0001'
    embedding = [1.0] + [0.0] * (EMBED_DIM - 1)
    item: dict[str, Any] = {
        'crop_id': item_id,
        'image_id': image_id,
        'image_path': f'/data/{slug}/{image_id}.jpg',
        'class_id': 1,
        'class_name': CLASS_NAMES[slug],
        'class_source': 'human',
        'validated': True,
        'bbox': [0.1, 0.1, 0.5, 0.5],
        'bbox_norm': [0.1, 0.1, 0.5, 0.5],
        'cluster_id': 1,
        'pe_embedding': embedding,
        'region_embedding': embedding,
    }
    # An unlabeled proposal: the batch label/move/region routes refuse to
    # overwrite a human decision, so they act on this one.
    proposal: dict[str, Any] = {
        **item,
        'crop_id': f'{slug}-item-0002',
        'class_id': None,
        'class_name': None,
        'class_source': 'proposal',
        'validated': False,
    }
    return {
        'items': {item_id: item, proposal['crop_id']: proposal},
        'images': {image_id: {'image_id': image_id, 'image_path': item['image_path']}},
        'labels_confirmed': {
            f'{slug}-label-0001': {
                'crop_id': item_id,
                'class_id': 1,
                'class_name': CLASS_NAMES[slug],
            }
        },
        'classes': {'1': {'class_id': 1, 'class_name': CLASS_NAMES[slug]}},
    }


# --- A recording layer with its own parser -----------------------------------


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
    parts = [unquote(p) for p in url.split('?')[0].split('/') if p]
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
            target = header.get('index')
            if target:
                out |= set(target.split(',') if isinstance(target, str) else target)
            elif not url_index:
                out.add('*')
    elif action in ('_search', '_count') and not url_index and parts[:2] != ['_search', 'scroll']:
        out.add('*')
    out.discard('')
    return out


class _FakeTransport:
    """The bottom of the fake: answers OpenSearch REST calls from a per-
    index doc store, and applies writes to it. Queries are not evaluated
    -- a search returns every doc of the index it names -- which is what a
    leak test wants: any index a route reaches shows up in its response."""

    def __init__(self, store: dict[str, dict[str, dict[str, Any]]]) -> None:
        from opensearchpy.serializer import JSONSerializer

        self.serializer = JSONSerializer()
        self.store = store
        self.writes: list[str] = []  # index names written
        self.received: list[set[str]] = []  # indexes of every request that got past the guard
        self._scrolls = 0
        self._tasks: dict[str, Any] = {}

    def _hits(self, indices: list[str]) -> list[dict[str, Any]]:
        return [
            {'_index': idx, '_id': doc_id, '_source': doc, '_seq_no': 1, '_primary_term': 1}
            for idx in indices
            for doc_id, doc in self.store.get(idx, {}).items()
        ]

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
        parts = [unquote(p) for p in url.split('?')[0].split('/') if p]
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
        if action in ('_search', '_msearch'):
            hits = self._hits(indices)
            resp: dict[str, Any] = {
                'hits': {'total': {'value': len(hits), 'relation': 'eq'}, 'hits': hits},
                'aggregations': {},
            }
            if (params or {}).get('scroll'):
                self._scrolls += 1
                resp['_scroll_id'] = f'scroll-{self._scrolls}'
            if action == '_msearch':
                return {'responses': [resp]}
            return resp
        if action == '_count':
            return {'count': len(self._hits(indices))}
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


# --- Fixture ------------------------------------------------------------------


def _record(slug: str, resources: Any) -> Any:
    from src.config.projects import ProjectRecord

    now = datetime.now(UTC).isoformat()
    return ProjectRecord(
        slug=slug,
        display_name=slug.title(),
        description='',
        status='active',
        revision=1,
        created_at=now,
        updated_at=now,
        origin=None,
        resources=resources,
    )


def _job_dirs(slug: str, tmp_path: Path) -> list[Path]:
    """The per-project job dirs P1 routes through ``project_jobs_dir``."""
    roots = [tmp_path / 'jobs' / name for name in ('probe', 'select', 'scores', 'viz')]
    return roots if slug == 'default' else [root / 'projects' / slug for root in roots]


def _seed_files(record: Any, tmp_path: Path) -> list[Path]:
    """Seed ``record``'s filesystem state; return the dirs that are its own
    (so a later pass can check they stay unchanged)."""
    from src.config.project_context import bind_project
    from src.services.detection.fp_store import FalsePositiveCentroidStore

    slug = record.slug
    res = record.resources
    res.class_registry_path.parent.mkdir(parents=True, exist_ok=True)
    res.class_registry_path.write_text(
        json.dumps(
            {
                'version': 1,
                'classes': [
                    {'id': 1, 'name': CLASS_NAMES[slug]},
                    {'id': 2, 'name': f'{slug}_second'},
                ],
            }
        ),
        encoding='utf-8',
    )
    state = {
        'job_id': f'{slug}-job-state',
        'status': 'completed',
        'result': {'crop_ids': [f'{slug}-item-0001']},
    }
    own_dirs = list(_job_dirs(slug, tmp_path))
    for job_dir in own_dirs:
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / 'state.json').write_text(json.dumps(state), encoding='utf-8')
    res.train_jobs_dir.mkdir(parents=True, exist_ok=True)
    (res.train_jobs_dir / f'{slug}-job-0001.status.json').write_text(
        json.dumps(
            {'job_id': f'{slug}-job-0001', 'state': 'finished', 'classes': [CLASS_NAMES[slug]]}
        ),
        encoding='utf-8',
    )
    res.autolabel_dir.mkdir(parents=True, exist_ok=True)
    (res.autolabel_dir / 'state.json').write_text(
        json.dumps({'job_id': f'{slug}-job-autolabel', 'status': 'completed'}), encoding='utf-8'
    )
    with bind_project(record):
        # The same ``trained_at`` in every project: a cache keyed only by
        # it would serve one project's scored crops to another.
        FalsePositiveCentroidStore().save(
            np.eye(1, EMBED_DIM, dtype=np.float32),
            {'trained_at': '2026-01-01T00:00:00Z', 'subids': [f'{slug}-fp-0001']},
        )
    for sub in ('exports', 'uploads'):
        (res.export_root if sub == 'exports' else res.upload_root).mkdir(
            parents=True, exist_ok=True
        )
    if slug == 'default':
        # default's dirs are the deployment roots every other project nests
        # under; only its own leaf files are checked.
        return [*own_dirs, res.class_registry_path.parent]
    return [
        *own_dirs,
        res.project_state_dir,
        res.train_jobs_dir,
        res.autolabel_dir,
        res.class_registry_path.parent,
    ]


@dataclass
class LeakEnv:
    app: Any
    records: dict[str, Any]
    accesses: list[tuple[str | None, str, str, str]]
    transport: _FakeTransport
    own_dirs: dict[str, list[Path]]
    events: list[dict[str, Any]] = field(default_factory=list)


@pytest.fixture
def leak_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reference_region_profile: None
) -> Any:
    """The real app, three seeded projects, a recording layer above the
    real guard, and no way out of the process (no network, no subprocess)."""
    import src.config.curation as curation_config_mod
    from src.clients import curation_opensearch
    from src.config.curation import IndexRole, base_curation_config
    from src.config.projects import resources_for_new
    from src.core.dependencies import app_state, get_async_triton
    from src.routers.curation import _common
    from src.services.curation import event_hub
    from src.services.curation.autolabel import job as autolabel_job
    from src.services.projects import guard, registry as registry_mod
    from src.services.projects.registry import ProjectRegistry, default_project_record

    for name, sub in {
        'STATE_DIR': 'state',
        'EXPORT_ROOT': 'default/exports',
        'UPLOAD_ROOT': 'default/uploads',
        'REGISTRY_PATH': 'default/class_registry.json',
        'BAKEOFF_EVAL_ROOT': 'default/bakeoff_eval',
        'CROP_CACHE_DIR': 'crop_cache',
        'TRAIN_JOBS_DIR': 'jobs/train',
        'AUTO_LABEL_STATE_DIR': 'jobs/auto_label',
        'BAKEOFF_JOBS_DIR': 'state/bakeoff_jobs',
        'PROJECTS_DATA_ROOT': 'projects',
        'PROBE_JOBS_DIR': 'jobs/probe',
        'SELECT_JOBS_DIR': 'jobs/select',
        'SCORES_STATE_DIR': 'jobs/scores',
        'VIZ_JOBS_DIR': 'jobs/viz',
        'REGION_DRAIN_STATE_DIR': 'jobs/region_drain',
        'HEARTBEAT_DIR': 'state/heartbeats',
        'TRAIN_RUNS_ROOT': 'state/training_runs',
        'BAKEOFF_OUT_DIR': 'state/bakeoff_out',
    }.items():
        monkeypatch.setenv(f'OP_{name}', str(tmp_path / sub))
    monkeypatch.setenv('OP_EVENT_BUS', 'process')
    monkeypatch.setenv('OP_SCORES_ENABLED', '1')
    monkeypatch.setenv('OP_REGION_FIELD_EMBEDDING', 'pe_embedding')
    monkeypatch.setattr(curation_config_mod, '_default_curation_config', None)
    monkeypatch.setattr(curation_opensearch, '_registries', {})
    monkeypatch.setattr(_common, '_INDEXES_BOOTSTRAPPED', set())
    # Keyed by index (audited), but its 5 s TTL would let wall-clock time
    # change which index roles a route reaches between the two passes.
    monkeypatch.setattr(curation_opensearch, '_SETTINGS_CACHE_TTL_SECONDS', 0.0)
    # Import-time constants of the (P2-owned) auto-label module: keep them
    # inside tmp_path so the sweep never touches the host's /jobs.
    al_dir = tmp_path / 'jobs' / 'auto_label'
    for attr, fname in {
        '_STATE_DIR': '',
        '_STATE_FILE': 'state.json',
        '_CANCEL_FLAG': 'cancel.flag',
        '_RUNNING_LOCK': 'running.lock',
        '_EXIT_CODE_FILE': 'exit_code',
        '_TRIGGER_FILE': 'trigger.json',
        '_HEARTBEAT_FILE': 'heartbeat',
    }.items():
        monkeypatch.setattr(autolabel_job, attr, al_dir / fname if fname else al_dir)

    base = base_curation_config()
    records = {'default': default_project_record()}
    for slug in ('alpha', 'beta'):
        records[slug] = _record(slug, resources_for_new(slug, base))

    store: dict[str, dict[str, dict[str, Any]]] = {}
    for slug, record in records.items():
        for role_value, docs in _docs(slug).items():
            store[record.resources.indexes[IndexRole(role_value)]] = docs
    own_dirs = {slug: _seed_files(record, tmp_path) for slug, record in records.items()}

    registry = ProjectRegistry(lambda: None)
    registry._by_slug = {s: r for s, r in records.items() if s != 'default'}
    registry._revision = 1

    async def _fresh(self: Any) -> None:
        return None

    monkeypatch.setattr(ProjectRegistry, 'ensure_fresh', _fresh)
    registry_mod.set_project_registry(registry)

    accesses: list[tuple[str | None, str, str, str]] = []
    fake = _FakeTransport(store)
    raw = AsyncOpenSearch(hosts=['http://127.0.0.1:9'])
    raw.transport = fake  # type: ignore[assignment]
    guard.install_project_guard(raw, registry)
    raw.transport = _RecordingTransport(raw.transport, accesses)  # type: ignore[assignment]

    from src.clients.opensearch import OpenSearchClient

    wrapper = OpenSearchClient(hosts=['http://127.0.0.1:9'])
    wrapper.client = raw
    monkeypatch.setattr(app_state, '_opensearch_client', wrapper)

    env = LeakEnv(app=None, records=records, accesses=accesses, transport=fake, own_dirs=own_dirs)
    monkeypatch.setattr(event_hub, '_HUB', None)
    hub = event_hub.get_event_hub()
    real_dispatch = hub._dispatch

    def _recording_dispatch(event: dict[str, Any]) -> None:
        env.events.append(dict(event))
        real_dispatch(event)

    monkeypatch.setattr(hub, '_dispatch', _recording_dispatch)

    def _no_network(*_a: Any, **_k: Any) -> Any:
        raise httpx.ConnectError('network disabled in the leak test')

    monkeypatch.setattr(httpx.AsyncHTTPTransport, 'handle_async_request', _no_network)
    monkeypatch.setattr(httpx.HTTPTransport, 'handle_request', _no_network)

    def _no_subprocess(*_a: Any, **_k: Any) -> Any:
        raise OSError('subprocesses disabled in the leak test')

    monkeypatch.setattr(subprocess, 'Popen', _no_subprocess)
    monkeypatch.setattr(subprocess, 'run', _no_subprocess)

    import asyncio

    async def _no_exec(*_a: Any, **_k: Any) -> Any:
        raise OSError('subprocesses disabled in the leak test')

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', _no_exec)
    monkeypatch.setattr(asyncio, 'create_subprocess_shell', _no_exec)

    class _DeadTriton:
        async def is_server_live(self) -> bool:
            return False

        def __getattr__(self, name: str) -> Any:
            raise ConnectionError(f'triton disabled in the leak test ({name})')

    from src.main import app

    env.app = app
    app.dependency_overrides[get_async_triton] = lambda: _DeadTriton()
    try:
        yield env
    finally:
        app.dependency_overrides.pop(get_async_triton, None)
        registry_mod.set_project_registry(None)


# --- The sweep ---------------------------------------------------------------


def _served_urls(value: Any) -> list[str]:
    """Every string in a JSON body that *is* a curation URL."""
    if isinstance(value, str):
        return [value] if value.startswith(f'{API}/') else []
    if isinstance(value, dict):
        return [u for v in value.values() for u in _served_urls(v)]
    if isinstance(value, list):
        return [u for v in value for u in _served_urls(v)]
    return []


def _scoped_routes(app: Any) -> list[tuple[str, str]]:
    """Every (method, path template) mounted under the scoped prefix."""
    out: list[tuple[str, str]] = []
    for route in app.routes:
        path = getattr(route, 'path', '')
        if not path.startswith(SCOPED):
            continue
        out.extend((method, path) for method in sorted(route.methods or ()) if method != 'HEAD')
    return out


def _fill(path: str, slug: str) -> str:
    params = route_params(slug)

    def _sub(match: re.Match[str]) -> str:
        name = match.group(1)
        if name == 'project':
            return slug
        if name not in params:
            raise AssertionError(f'unmapped route {path}: add {name!r} to route_params')
        return params[name]

    return re.sub(r'\{(\w+)(?::\w+)?\}', _sub, path)


def _is_streaming(app: Any, path: str) -> bool:
    for route in app.routes:
        if getattr(route, 'path', '') == path:
            annotation = inspect.signature(route.endpoint).return_annotation
            return 'StreamingResponse' in str(annotation)
    return False


def _markers(slug: str) -> tuple[str, ...]:
    return (
        *(f'{slug}{kind}' for kind in ('-item', '-img', '-label', '-job', '-fp', '-campaign')),
        CLASS_NAMES[slug],
    )


def _dir_digest(dirs: list[Path]) -> dict[str, str]:
    digest: dict[str, str] = {}
    for root in dirs:
        if root.is_file():
            digest[str(root)] = hashlib.sha256(root.read_bytes()).hexdigest()
            continue
        if not root.exists():
            continue
        for path in sorted(root.rglob('*')):
            if path.is_file():
                digest[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def _sweep(
    env: LeakEnv, slug: str
) -> tuple[list[str], set[tuple[str, str]], dict[tuple[str, str], frozenset[str]]]:
    """Call every scoped route as ``slug``. Returns (leaks, routes that
    wrote, the index roles each route reached)."""
    app, records = env.app, env.records
    own_indexes = set(records[slug].resources.indexes.values())
    foreign_indexes = {
        name: other
        for other, record in records.items()
        if other != slug
        for name in record.resources.indexes.values()
    }
    foreign_markers = [m for other in SLUGS if other != slug for m in _markers(other)]
    bodies = route_bodies(slug)
    own_dir_list = env.own_dirs[slug]

    role_of = {name: role.value for role, name in records[slug].resources.indexes.items()}
    leaks: list[str] = []
    wrote: set[tuple[str, str]] = set()
    roles: dict[tuple[str, str], frozenset[str]] = {}
    client = TestClient(app, raise_server_exceptions=False)
    # Reads first, against the seeded state; then every write.
    ordered = sorted(_scoped_routes(app), key=lambda route: route[0] != 'GET')
    for method, template in ordered:
        if template in STREAMING_ROUTES:
            continue
        key = (method, template[len(SCOPED) :])
        deferred = key in P2_DEFERRED
        url = _fill(template, slug)
        before_access, before_write, before_events = (
            len(env.accesses),
            len(env.transport.writes),
            len(env.events),
        )
        own_before = _dir_digest(own_dir_list)
        kwargs = bodies.get(key, {} if method in ('GET', 'DELETE') else {'json': {}})
        response = client.request(method, url, **kwargs)
        route_accesses = env.accesses[before_access:]
        route_writes = env.transport.writes[before_write:]
        route_events = env.events[before_events:]
        tag = f'[{slug}] {method} {key[1]}'
        roles[key] = frozenset(role_of.get(index, 'foreign') for *_x, index in route_accesses)

        for bound, _verb, os_url, index in route_accesses:
            if index == '*' or index in foreign_indexes:
                leaks.append(
                    f'{tag}: bound={bound} reached {foreign_indexes.get(index, "every")!r} '
                    f'index {index} ({os_url})'
                )
            if bound != slug:
                leaks.append(f'{tag}: OpenSearch call bound to {bound!r}')
        leaks.extend(
            f'{tag}: event {event.get("type")} went to project {event.get("project")!r}'
            for event in route_events
            if event.get('project') != slug
        )

        body = response.text
        if not deferred:
            leaks.extend(f'{tag}: response carries {m!r}' for m in foreign_markers if m in body)
        if response.headers.get('content-type', '').startswith('application/json'):
            leaks.extend(
                f'{tag}: served URL {u!r} is not under {slug} prefix'
                for u in _served_urls(response.json())
                if u != f'{API}/projects/{slug}' and not u.startswith(f'{API}/projects/{slug}/')
            )
        if response.status_code >= 500 and key not in EXPECTED_5XX and not deferred:
            leaks.append(f'{tag}: unexpected {response.status_code} {body[:300]}')
        if method != 'GET':
            if response.status_code == 422 and key not in NO_WRITE:
                leaks.append(f'{tag}: unmapped body (422 {body[:200]})')
            if (
                any(w in own_indexes for w in route_writes)
                or _dir_digest(own_dir_list) != own_before
            ):
                wrote.add(key)
            elif key not in NO_WRITE and not deferred:
                leaks.append(
                    f'{tag}: mutating route wrote nothing ({response.status_code} {body[:200]})'
                )
    return leaks, wrote, roles


@pytest.mark.parametrize(('first', 'second'), [('alpha', 'beta'), ('beta', 'alpha')])
def test_every_scoped_route_stays_inside_the_bound_project(
    leak_env: LeakEnv, first: str, second: str
) -> None:
    app = leak_env.app
    routes = _scoped_routes(app)
    assert len(routes) > 100, f'expected the full scoped surface, got {len(routes)} routes'
    streaming_unmapped = sorted({p for _m, p in routes if _is_streaming(app, p)} - STREAMING_ROUTES)
    assert not streaming_unmapped, f'unmapped streaming route(s): {streaming_unmapped}'
    mutating = {(m, p[len(SCOPED) :]) for m, p in routes if m != 'GET'}
    stale = sorted((set(NO_WRITE) | set(route_bodies('x'))) - mutating)
    assert not stale, f'NO_WRITE/route_bodies entries for routes that no longer exist: {stale}'

    leaks_first, _, roles_first = _sweep(leak_env, first)
    first_dirs = _dir_digest(leak_env.own_dirs[first])
    leaks_second, wrote, roles_second = _sweep(leak_env, second)
    # A process cache keyed without the project answers the second project
    # from the first one's data, so the route skips its own query: the
    # same route, as the other project, must reach the same index roles.
    skipped = [
        f'[{second}] {m} {p}: reached {sorted(roles_second[(m, p)])}, '
        f'but as {first} reached {sorted(roles_first[(m, p)])} (unkeyed cache?)'
        for (m, p) in roles_first
        if roles_first[(m, p)] != roles_second.get((m, p)) and (m, p) not in P2_DEFERRED
    ]
    changed = sorted(
        path
        for path, digest in _dir_digest(leak_env.own_dirs[first]).items()
        if first_dirs.get(path) != digest
    ) + sorted(set(first_dirs) - set(_dir_digest(leak_env.own_dirs[first])))

    leaks = leaks_first + leaks_second + skipped
    leaks.extend(f"[{second}] changed {first}'s file {p}" for p in changed)
    assert not leaks, 'cross-project leak(s):\n' + '\n'.join(sorted(set(leaks)))
    # Not vacuous: every mutating route not excused in NO_WRITE really wrote.
    expected_writers = mutating - set(NO_WRITE) - set(P2_DEFERRED)
    assert expected_writers <= wrote, f'routes that never wrote: {sorted(expected_writers - wrote)}'
    stale_excuses = sorted(set(NO_WRITE) & wrote)
    assert not stale_excuses, f'these routes do write; drop them from NO_WRITE: {stale_excuses}'
    assert len(wrote) >= 20, f'only {len(wrote)} mutating routes wrote to {second}'


def test_a_misrouted_mget_is_refused_before_it_reaches_opensearch(
    leak_env: LeakEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A planted scoping bug: ``mget_crops`` aimed at alpha's items index
    while beta is bound. opensearch-py sends ``mget``'s body as a dict; the
    guard must still read it and refuse, so alpha's documents never leave
    OpenSearch."""
    from src.clients import curation_opensearch
    from src.config.curation import IndexRole

    alpha_items = leak_env.records['alpha'].resources.indexes[IndexRole.ITEMS]
    real_mget_crops = curation_opensearch.mget_crops

    async def _misrouted(client: Any, crop_ids: Any, **kwargs: Any) -> Any:
        kwargs['index'] = alpha_items
        return await real_mget_crops(client, crop_ids, **kwargs)

    monkeypatch.setattr(curation_opensearch, 'mget_crops', _misrouted)
    before = len(leak_env.transport.received)
    client = TestClient(leak_env.app, raise_server_exceptions=False)
    response = client.put(
        f'{API}/projects/beta/crops/batch_label',
        json={'crop_ids': ['alpha-item-0001'], 'class_id': 1},
    )
    reached = set().union(*leak_env.transport.received[before:])
    assert alpha_items not in reached, "the misrouted mget reached alpha's index"
    assert response.status_code == 500
    assert 'internal_isolation_error' in response.text


@pytest.mark.parametrize('injected', ['alpha', None])
def test_publish_cannot_redirect_an_event(leak_env: LeakEnv, injected: str | None) -> None:
    """``POST .../beta/events/publish`` with ``extra.project`` naming another
    project (or the global stream) is refused, and nothing is delivered
    anywhere but beta's stream."""
    client = TestClient(leak_env.app, raise_server_exceptions=False)
    before = len(leak_env.events)
    response = client.post(
        f'{API}/projects/beta/events/publish',
        json={
            'type': 'crop.classified',
            'crop_id': 'beta-item-0001',
            'extra': {'project': injected},
        },
    )
    assert response.status_code == 422, response.text
    assert all(event.get('project') == 'beta' for event in leak_env.events[before:])
