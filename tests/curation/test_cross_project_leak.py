"""The isolation proof (projects_plan.md §10 P1): every scoped curation
route, swept first as one project and then as the other (alpha then beta,
and beta then alpha), reads and writes only the bound project's
OpenSearch indexes, directories, caches and event stream.

Why two passes: a process cache primed by the first project is read back
by the second one. A single sweep as one project can never see that.

Setup: three ordinary projects -- ``default``, ``alpha`` and ``beta`` --
on a fake OpenSearch transport behind the real
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

import base64
import collections
import hashlib
import inspect
import json
import os
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
from integration.ingest_fakes import FakePEEncoder, FakeTritonPool, jpeg_bytes
from opensearchpy import AsyncOpenSearch
from opensearchpy.exceptions import NotFoundError

from curation.query_fakes import _aggregate, matches


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
    # Ingest takes a server path under the source root; import matches
    # the stored (relative) image_path of an already-ingested image.
    source = f'{_source_root()}/{slug}'
    img = f'{source}/{slug}-new-0001.jpg'
    labeled = f'{source}/{slug}-lab-0001.jpg'
    b64 = base64.b64encode(jpeg_bytes(len(slug))).decode('ascii')
    force = {'force': 'true'}
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
        ('POST', '/crops/label/undo_batch'): {'json': {'crop_ids': [f'{slug}-item-0003']}},
        ('POST', '/crops/{crop_id}/discard'): {'json': {}},
        ('POST', '/crops/discard_batch'): {'json': {'crop_ids': [item]}},
        ('POST', '/crops/region/undo_batch'): {'json': {'crop_ids': [f'{slug}-item-0006']}},
        ('POST', '/events/publish'): {
            'json': {'type': 'crop.classified', 'crop_id': item, 'class_id': 1}
        },
        ('POST', '/export/yolo'): {'json': {'version_tag': f'{slug}-v1'}},
        ('POST', '/export/single_class'): {'json': {'version_tag': f'{slug}-v1', 'class_ids': [1]}},
        ('POST', '/vlm/label_batch'): {'json': {'crop_ids': [item]}},
        ('POST', '/vlm/verify_regions'): {'json': {'crop_ids': [f'{slug}-item-0004']}},
        ('POST', '/vlm/verify_region_batch'): {
            'json': {'items': [{'crop_id': item, 'region_image_b64': b64}]}
        },
        ('POST', '/vlm/region_visible_batch'): {
            'json': {'items': [{'crop_id': item, 'image_b64': b64}]}
        },
        ('POST', '/ingest/image'): {'json': {'path': img}},
        ('POST', '/ingest/batch'): {'json': {'items': [{'path': f'{source}/{slug}-new-0002.jpg'}]}},
        ('POST', '/import_labels'): {
            'json': {'image_path': labeled, 'label_txt_path': labeled.replace('.jpg', '.txt')}
        },
        ('POST', '/import_labels/batch'): {
            'json': {
                'items': [
                    {'image_path': labeled, 'label_txt_path': labeled.replace('.jpg', '.txt')}
                ]
            }
        },
        ('POST', '/ingest/path_lookup'): {'json': {'image_paths': [img]}},
        ('POST', '/ingest/upload'): {
            'files': [('images', (f'{slug}-up.jpg', jpeg_bytes(len(slug)), 'image/jpeg'))],
        },
        ('POST', '/probe/run'): {'json': {'job_id': f'{slug}-job-0001'}},
        ('PUT', '/models/{model_name}/sharing'): {'json': {'shared': True, 'expected_revision': 1}},
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
        # force: the preflight's class-balance/disk gates are not what
        # this test is about; the job files written are.
        ('POST', '/train/start'): {'params': force, 'json': {}},
        ('POST', '/train/start_campaign'): {
            'params': force,
            'json': {
                'campaign_id': f'{slug}-campaign-0001',
                'dataset_export_dir': f'/exports/{slug}-v1',
                'runs': [{'profile': 'probe', 'model_size': 'n'}],
            },
        },
        ('POST', '/train/promote/{job_id}'): {
            'json': {'triton_name': f'{slug}_model', 'force': True}
        },
    }


# Mutating routes that write nothing by design (read-only work under POST,
# or an event with no data write), and why. Every other mutating route must
# really write in the sweep.
NO_WRITE: dict[tuple[str, str], str] = {
    ('POST', '/events/publish'): 'publishes an event (checked separately), writes no data',
    ('POST', '/ingest/path_lookup'): 'read-only lookup under POST',
    ('POST', '/train/preflight'): 'read-only validation under POST',
    ('POST', '/train/reload_promoted'): 'asks Triton to load promoted models; stores nothing',
    (
        'POST',
        '/vlm/verify_region_batch',
    ): "returns the VLM's verdicts to the caller; stores nothing",
    (
        'POST',
        '/vlm/region_visible_batch',
    ): "returns the VLM's verdicts to the caller; stores nothing",
}

# Routes whose ``{crop_id}`` is a seeded item other than ``-item-0001``,
# because they act on state that item does not have.
CROP_FOR: dict[tuple[str, str], str] = {
    ('POST', '/crops/{crop_id}/vlm_dismiss'): 'item-0004',
    ('POST', '/crops/{crop_id}/vlm_dismiss/undo'): 'item-0004',
    ('POST', '/crops/{crop_id}/region/undo'): 'item-0005',
}

# Routes that answer 5xx in the fixture for a reason that is not isolation.
EXPECTED_5XX: dict[tuple[str, str], str] = {}

# Known leaks owned by P2 (cutover/projects-workers, projects_plan.md §5):
# process-global state P1 did not create and P2 makes per project. Each
# entry: (method, template) -> the foreign-project evidence it may show.
# Anything else is a P1 failure.
P2_DEFERRED: dict[tuple[str, str], str] = {
    ('POST', '/pipeline/auto_label/start'): 'global auto-label trigger/state dir (P2)',
    ('POST', '/pipeline/auto_label'): 'global auto-label trigger/state dir (P2)',
    ('GET', '/pipeline/auto_label/status'): 'global auto-label state dir (P2)',
    ('GET', '/train/runs'): 'global training staging dir (P2)',
    ('GET', '/train/status'): 'global training staging dir (P2)',
    ('POST', '/vlm/label_cluster/{cluster_id}'): 'queues the global auto-label job (P2)',
    ('POST', '/pipeline/auto_label/cancel'): 'global auto-label state dir (P2)',
    ('POST', '/bakeoff/run'): 'global bake-off jobs dir and GPU claim (P2, plan §5.3)',
    ('POST', '/train/promote/{job_id}'): 'promoted-model ownership in the shared Triton repo (P2)',
    ('DELETE', '/models/{model_name}'): 'promoted-model ownership in the shared Triton repo (P2)',
    ('PUT', '/models/{model_name}/sharing'): (
        'promoted-model ownership in the shared Triton repo (P2); no promote.json seeded here'
    ),
}


def _running_job(job_dir: str) -> Any:
    """A cancel acts on a running job: mark ``slug``'s one running (a
    fresh state, no heartbeat yet) right before its cancel is called."""

    def prepare(env: LeakEnv, slug: str) -> None:
        state = env.root / 'jobs' / job_dir / 'projects' / slug / 'state.json'
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_text(
            json.dumps({'job_id': f'{slug}-job-running', 'status': 'running'}), encoding='utf-8'
        )
        (state.parent / 'heartbeat').unlink(missing_ok=True)

    return prepare


PREPARE: dict[tuple[str, str], Any] = {
    ('POST', '/probe/cancel'): _running_job('probe'),
    ('POST', '/scores/cancel'): _running_job('scores'),
    ('POST', '/select/cancel'): _running_job('select'),
    ('POST', '/viz/projection/cancel'): _running_job('viz'),
}


# Routes that read the trainer's jobs dir, which is still shared across
# projects until P2 scopes it (plan §5.3): another project's job ids may
# show in their responses. They must still write (and stay off every
# other project's indexes and dirs).
P2_SHARED_TRAIN_JOBS: dict[tuple[str, str], str] = {
    ('GET', '/bakeoff/trained_models'): 'lists every run in the shared trainer jobs dir (P2)',
    ('POST', '/train/preflight'): 'the active-run check scans the shared trainer jobs dir (P2)',
    ('POST', '/train/start'): 'the active-run check scans the shared trainer jobs dir (P2)',
}


# Long-lived SSE streams: the per-project delivery they serve is proven by
# the event checks below and tests/projects/test_event_hub_project_filter.py.
STREAMING_ROUTES: frozenset[str] = frozenset({f'{SCOPED}/events', f'{SCOPED}/pipeline/events'})


def _source_root() -> str:
    """The shared source-image root (``OP_SOURCE_ROOT``, set by the fixture)."""
    return os.environ.get('OP_SOURCE_ROOT', '/images')


def _vec(i: int) -> list[float]:
    """A distinct unit embedding per seeded item (the same across projects,
    so both passes see the same data shapes)."""
    v = np.random.default_rng(i).normal(size=EMBED_DIM)
    return (v / np.linalg.norm(v)).tolist()


def _docs(slug: str) -> dict[str, dict[str, dict[str, Any]]]:
    """One project's seed data, keyed by IndexRole value -> doc id -> doc.

    Beyond a validated item and an unlabeled proposal, it seeds the state
    each write route acts on (so every one of them really writes): a
    human label to undo, a VLM suggestion to dismiss, region edits to
    undo, a region to verify, a high-purity candidate cluster to promote,
    a pending new-class proposal, a residual pool to refit UMAP on, and
    an ingested image whose label file can be imported."""
    from src.config.region_fields import get_region_fields
    from src.services.curation.class_sources import VLM_NEW_CLASS_PENDING_CLASS_SOURCE
    from src.services.curation.edit_history import EDIT_HISTORY_FIELD, EditKind, record_edit
    from src.services.curation.history import record_class_snapshot

    F = get_region_fields()
    image_id = f'{slug}-img-0001'

    def crop(n: int, **fields: Any) -> dict[str, Any]:
        return {
            'crop_id': f'{slug}-item-{n:04d}',
            'image_id': image_id,
            'image_path': f'{slug}/{image_id}.jpg',
            'class_id': 1,
            'class_name': CLASS_NAMES[slug],
            'class_source': 'human',
            'class_validated': True,
            'bbox': [0.1, 0.1, 0.5, 0.5],
            'bbox_norm': [0.1, 0.1, 0.5, 0.5],
            'cluster_id': 1,
            'pe_embedding': _vec(n),
            'region_embedding': _vec(n),
            **fields,
        }

    unlabeled = {
        'class_id': None,
        'class_name': None,
        'class_source': 'proposal',
        'class_validated': False,
    }
    region_before = {F.bbox_norm: None}
    items = [
        crop(1),
        # An unlabeled proposal: the batch label/move/region routes refuse
        # to overwrite a human decision, so they act on this one.
        crop(2, **unlabeled),
        crop(
            3,
            class_id_history=record_class_snapshot(
                crop(3, **unlabeled), writer='human:label_crop', restorable=True
            ),
        ),
        crop(
            4,
            class_source='vlm',
            class_validated=False,
            vlm_confidence='high',
            **{F.bbox_norm: [0.2, 0.2, 0.3, 0.3]},
        ),
        *(
            crop(
                n,
                **{
                    F.bbox_norm: [0.2, 0.2, 0.3, 0.3],
                    EDIT_HISTORY_FIELD: record_edit(
                        region_before, kind=EditKind.REGION, writer='human:set_region'
                    ),
                },
            )
            for n in (5, 6)
        ),
        *(
            crop(n, class_source='vlm', class_validated=False, cluster_id=10001)
            for n in range(7, 11)
        ),
        crop(
            11,
            **{
                **unlabeled,
                'class_source': VLM_NEW_CLASS_PENDING_CLASS_SOURCE,
                'vlm_proposed_class': f'{slug}-proposal',
            },
        ),
        *(crop(n, **unlabeled, cluster_id=None) for n in range(12, 18)),
    ]
    labeled_image = f'{slug}-img-0002'
    return {
        'items': {doc['crop_id']: doc for doc in items},
        'images': {
            image_id: {'image_id': image_id, 'image_path': f'{slug}/{image_id}.jpg'},
            labeled_image: {
                'image_id': labeled_image,
                'image_path': f'{_source_root()}/{slug}/{slug}-lab-0001.jpg',
            },
        },
        'labels_confirmed': {
            f'{slug}-label-0001': {
                'crop_id': f'{slug}-item-0001',
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
    roots = [
        tmp_path / 'jobs' / name for name in ('probe', 'select', 'scores', 'viz', 'region_drain')
    ]
    return [root / 'projects' / slug for root in roots]


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
    # Source images every project may ingest from (the shared source root).
    images = tmp_path / 'images' / slug
    images.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(('img-0001', 'new-0001', 'new-0002', 'lab-0001')):
        (images / f'{slug}-{name}.jpg').write_bytes(jpeg_bytes(i + len(slug)))
    (images / f'{slug}-lab-0001.txt').write_text('1 0.5 0.5 0.4 0.4\n', encoding='utf-8')
    own_dirs = list(_job_dirs(slug, tmp_path))
    for job_dir in own_dirs:
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / 'state.json').write_text(json.dumps(state), encoding='utf-8')
    # A finished training run with its checkpoint, in the project's own
    # jobs dir and (until P2 scopes the trainer, plan §5.3) in the shared
    # one the training service still reads.
    checkpoint = tmp_path / 'state' / 'training_runs' / slug / 'weights' / 'best.pt'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b'fake checkpoint')
    finished = {
        'job_id': f'{slug}-job-0001',
        'state': 'finished',
        'classes': [CLASS_NAMES[slug]],
        'checkpoint_path': str(checkpoint),
    }
    for jobs_dir in (res.train_jobs_dir, tmp_path / 'jobs' / 'train'):
        jobs_dir.mkdir(parents=True, exist_ok=True)
        (jobs_dir / f'{slug}-job-0001.status.json').write_text(
            json.dumps(finished), encoding='utf-8'
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
    return [
        *own_dirs,
        res.project_state_dir,
        res.train_jobs_dir,
        res.autolabel_dir,
        res.class_registry_path.parent,
    ]


class _LeakPEEncoder(FakePEEncoder):
    """The ingest fake, at the seed data's embedding width (so the
    clustering and scoring routes see one consistent pool)."""

    async def embed_crops(self, crops: list[np.ndarray], max_batch: int = 32) -> np.ndarray:  # noqa: ARG002
        return np.array([_vec(100 + i) for i in range(len(crops))], dtype=np.float32)

    async def embed_whole_frame(self, path: str) -> np.ndarray | None:
        self.whole_frame_paths.append(path)
        return np.array(_vec(200), dtype=np.float32)

    async def embed_whole_frame_bytes(self, data: bytes) -> np.ndarray | None:
        self.whole_frame_bytes.append(data)
        return np.array(_vec(201), dtype=np.float32)


@dataclass
class LeakEnv:
    app: Any
    records: dict[str, Any]
    accesses: list[tuple[str | None, str, str, str]]
    transport: _FakeTransport
    own_dirs: dict[str, list[Path]]
    root: Path
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
    from src.config.projects import new_project_record, resources_for_new
    from src.core.dependencies import app_state, get_async_triton
    from src.routers.curation import _common
    from src.services.curation import event_hub
    from src.services.projects import guard, registry as registry_mod
    from src.services.projects.registry import ProjectRegistry

    for name, sub in {
        'STATE_DIR': 'state',
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
        'SOURCE_ROOT': 'images',
        'TRITON_MODEL_REPO': 'models',
    }.items():
        monkeypatch.setenv(f'OP_{name}', str(tmp_path / sub))
    monkeypatch.setenv('OP_EVENT_BUS', 'process')
    for flag in ('SCORES_ENABLED', 'SELECT_DIVERSE_ENABLED', 'VIZ_PROJECTION_ENABLED'):
        monkeypatch.setenv(f'OP_{flag}', '1')
    # Diverse selection over the job path (the one that writes); the
    # inline path answers from memory.
    monkeypatch.setenv('OP_SELECT_SYNC_MAX_OPS', '1')
    monkeypatch.setenv('OP_REGION_FIELD_EMBEDDING', 'pe_embedding')
    monkeypatch.setenv('OP_INGEST_PRIMARY_DETECTOR_MODEL', 'fake_item_detector')
    monkeypatch.setattr(curation_config_mod, '_default_curation_config', None)
    monkeypatch.setattr(curation_opensearch, '_registries', {})
    monkeypatch.setattr(_common, '_INDEXES_BOOTSTRAPPED', set())
    # Keyed by index (audited), but its 5 s TTL would let wall-clock time
    # change which index roles a route reaches between the two passes.
    monkeypatch.setattr(curation_opensearch, '_SETTINGS_CACHE_TTL_SECONDS', 0.0)
    # A UMAP fit the fixture's small pool supports (the defaults need
    # more points than the fixture seeds).
    from src.services.curation.clustering import embedding_reduce

    monkeypatch.setattr(embedding_reduce, 'UMAP_N_COMPONENTS', 2)
    monkeypatch.setattr(embedding_reduce, 'UMAP_N_NEIGHBORS', 3)
    # The (P2-owned) auto-label module resolves its state dir fresh per
    # bound project on every call (`_state_dir()` -> the bound project's
    # own `autolabel_dir`, a PROJECT_SCOPED_FIELDS entry) -- there is no
    # import-time module constant left to patch. Redirect the env var
    # `resources_for_new` reads instead, so the sweep never touches the
    # host's /jobs and each project's own nested dir
    # (`<this>/projects/<slug>`, `default` included per P1R §6.1/D-A) is
    # kept inside tmp_path.
    monkeypatch.setenv('OP_AUTO_LABEL_STATE_DIR', str(tmp_path / 'jobs' / 'auto_label'))

    base = base_curation_config()
    records = {'default': new_project_record('default', base)}
    for slug in ('alpha', 'beta'):
        records[slug] = _record(slug, resources_for_new(slug, base))

    store: dict[str, dict[str, dict[str, Any]]] = {}
    for slug, record in records.items():
        for role_value, docs in _docs(slug).items():
            store[record.resources.indexes[IndexRole(role_value)]] = docs
    own_dirs = {slug: _seed_files(record, tmp_path) for slug, record in records.items()}

    registry = ProjectRegistry(lambda: None)
    registry._by_slug = dict(records)
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

    env = LeakEnv(
        app=None,
        records=records,
        accesses=accesses,
        transport=fake,
        own_dirs=own_dirs,
        root=tmp_path,
    )
    monkeypatch.setattr(event_hub, '_HUB', None)
    hub = event_hub.get_event_hub()
    real_dispatch = hub._dispatch

    def _recording_dispatch(event: dict[str, Any]) -> None:
        env.events.append(dict(event))
        real_dispatch(event)

    monkeypatch.setattr(hub, '_dispatch', _recording_dispatch)

    def _no_network(*_a: Any, **_k: Any) -> Any:
        raise httpx.ConnectError('network disabled in the leak test')

    async def _only_the_fake_vlm(_self: Any, request: httpx.Request) -> httpx.Response:
        if request.url.host == _FAKE_VLM_HOST:
            return _fake_vlm_reply(request)
        return _no_network()

    monkeypatch.setattr(httpx.AsyncHTTPTransport, 'handle_async_request', _only_the_fake_vlm)
    monkeypatch.setattr(httpx.HTTPTransport, 'handle_request', _no_network)
    # Every VLM route talks to the in-process fake VLM (a real
    # OpenAI-shaped reply), so the labeling/verify routes really write.
    from src.routers.curation import vlm as vlm_router
    from src.services.labeling.vlm_labeler import VlmLabeler

    defaults = VlmLabeler.__init__.__defaults__ or ()
    monkeypatch.setattr(
        VlmLabeler.__init__, '__defaults__', (f'http://{_FAKE_VLM_HOST}/v1', *defaults[1:])
    )
    monkeypatch.setitem(vlm_router._get_vlm_labeler.__dict__, '_insts', {})

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

    import src.main as main_module
    from src.main import app

    env.app = app
    app.dependency_overrides[get_async_triton] = lambda: _DeadTriton()
    # Ingest runs end to end: a fake detector pool and PE encoder at the
    # service boundary (the same fakes the ingest integration tests use).
    monkeypatch.setattr(main_module, 'get_async_triton_pool', lambda: FakeTritonPool())
    had_encoder = hasattr(app.state, 'pe_encoder')
    previous_encoder = getattr(app.state, 'pe_encoder', None)
    app.state.pe_encoder = _LeakPEEncoder()
    try:
        yield env
    finally:
        app.dependency_overrides.pop(get_async_triton, None)
        registry_mod.set_project_registry(None)
        if had_encoder:
            app.state.pe_encoder = previous_encoder
        else:
            del app.state.pe_encoder


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


def _fill(path: str, slug: str, key: tuple[str, str] | None = None) -> str:
    params = route_params(slug)
    if key in CROP_FOR:
        params['crop_id'] = f'{slug}-{CROP_FOR[key]}'

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


Shape = tuple[str, str, str]  # (index role, HTTP verb, OpenSearch action)


def _shape(role_of: dict[str, str], verb: str, os_url: str, index: str) -> Shape:
    parts = [p for p in os_url.split('?', 1)[0].split('/') if p]
    action = next((p for p in parts if p.startswith('_')), '<index>')
    return role_of.get(index, 'foreign'), verb, action


def _sweep(
    env: LeakEnv, slug: str, only: Any = None
) -> tuple[list[str], set[tuple[str, str]], dict[tuple[str, str], collections.Counter[Shape]]]:
    """Call every scoped route (or those ``only`` accepts) as ``slug``.
    Returns (leaks, routes that wrote, the OpenSearch request shapes each
    route issued, counted per index role)."""
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

    role_of = {name: role.value for role, name in records[slug].resources.indexes.items()}
    leaks: list[str] = []
    wrote: set[tuple[str, str]] = set()
    roles: dict[tuple[str, str], collections.Counter[Shape]] = {}
    client = TestClient(app, raise_server_exceptions=False)
    # Reads first, against the seeded state; then every write.
    ordered = sorted(_scoped_routes(app), key=lambda route: route[0] != 'GET')
    for method, template in ordered:
        if template in STREAMING_ROUTES:
            continue
        key = (method, template[len(SCOPED) :])
        if only is not None and not only(key):
            continue
        deferred = key in P2_DEFERRED
        url = _fill(template, slug, key)
        before_access, before_write, before_events = (
            len(env.accesses),
            len(env.transport.writes),
            len(env.events),
        )
        if key in PREPARE:
            PREPARE[key](env, slug)
        tree_before = _dir_digest([env.root]) if method != 'GET' else {}
        kwargs = bodies.get(key, {} if method in ('GET', 'DELETE') else {'json': {}})
        response = client.request(method, url, **kwargs)
        route_accesses = env.accesses[before_access:]
        route_writes = env.transport.writes[before_write:]
        route_events = env.events[before_events:]
        tag = f'[{slug}] {method} {key[1]}'
        roles[key] = collections.Counter(
            _shape(role_of, verb, os_url, index) for _b, verb, os_url, index in route_accesses
        )

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
        if not deferred and key not in P2_SHARED_TRAIN_JOBS:
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
            # A write is an OpenSearch write to its own index or any file it
            # created/changed (another project's dirs are checked apart).
            if (
                any(w in own_indexes for w in route_writes)
                or _dir_digest([env.root]) != tree_before
            ):
                wrote.add(key)
            elif key not in NO_WRITE and not deferred:
                leaks.append(
                    f'{tag}: mutating route wrote nothing ({response.status_code} {body[:200]})'
                )
    return leaks, wrote, roles


def _cache_parity(
    first: str,
    second: str,
    shapes_first: dict[tuple[str, str], collections.Counter[Shape]],
    shapes_second: dict[tuple[str, str], collections.Counter[Shape]],
) -> list[str]:
    """A process cache keyed without the project answers the second project
    from the first one's data, so the route skips (some of) its own
    queries. Each route is the first call of its kind for each project, so
    the same route must issue the same requests, per index role and
    OpenSearch action, as either project."""
    return [
        f'[{second}] {m} {p}: issued {sorted(shapes_second.get((m, p), {}).items())}, '
        f'but as {first} {sorted(shapes_first[(m, p)].items())} (unkeyed cache?)'
        for (m, p) in shapes_first
        if shapes_first[(m, p)] != shapes_second.get((m, p)) and (m, p) not in P2_DEFERRED
    ]


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
    every = {(m, p[len(SCOPED) :]) for m, p in routes}
    mapped = set(NO_WRITE) | set(route_bodies('x')) | set(CROP_FOR) | set(PREPARE)
    stale = sorted((mapped - mutating) | ((set(P2_DEFERRED) | set(P2_SHARED_TRAIN_JOBS)) - every))
    assert not stale, f'entries for routes that no longer exist: {stale}'

    leaks_first, _, roles_first = _sweep(leak_env, first)
    first_dirs = _dir_digest(leak_env.own_dirs[first])
    leaks_second, wrote, roles_second = _sweep(leak_env, second)
    skipped = _cache_parity(first, second, roles_first, roles_second)
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


@pytest.mark.parametrize('event_type', ['project.created', 'combine.finished'])
def test_publish_refuses_global_event_types(leak_env: LeakEnv, event_type: str) -> None:
    """Re-review R11: a client cannot spoof a lifecycle/combine event, not
    even on its own project's stream."""
    client = TestClient(leak_env.app, raise_server_exceptions=False)
    before = len(leak_env.events)
    response = client.post(f'{API}/projects/beta/events/publish', json={'type': event_type})
    assert response.status_code == 422, response.text
    assert leak_env.events[before:] == []


def test_a_planted_unkeyed_cache_behind_a_shared_helper_is_caught(
    leak_env: LeakEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-review R3: an unkeyed cache around the review-sort coverage helper
    (``strategy_registry.field_coverage``) serves the second project the
    first one's coverage while the route still searches its own items
    index. The per-role request parity must catch it."""
    from src.services.curation import strategy_registry

    planted: dict[str, Any] = {}
    real = strategy_registry.field_coverage

    async def _unkeyed(opensearch: Any, fields: frozenset[str]) -> Any:
        if 'hit' not in planted:
            planted['hit'] = await real(opensearch, fields)
        return planted['hit']

    monkeypatch.setattr(strategy_registry, 'field_coverage', _unkeyed)

    def only(key: tuple[str, str]) -> bool:
        return key[0] == 'GET' and key[1].startswith('/review/')

    _, _, shapes_alpha = _sweep(leak_env, 'alpha', only)
    _, _, shapes_beta = _sweep(leak_env, 'beta', only)
    caught = _cache_parity('alpha', 'beta', shapes_alpha, shapes_beta)
    assert any('/review/' in line for line in caught), 'the planted unkeyed cache went unnoticed'
