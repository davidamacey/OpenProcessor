"""Curation text-search latency per stage (#414): p50/p95/p99, unfiltered and by selectivity band.

Drives ``GET /curation/projects/{project}/search/text`` with distinct real COCO captions (so the
per-worker text LRU never hits), discards a warm-up longer than the worker count (kept apart as
the ``cold`` rows), and reads the per-stage durations the route serves in its ``Server-Timing``
header. Filter bands are classes whose share of the project's embedded items is nearest to the
targets; the shares are measured from OpenSearch at the start and written into the result.
A short repeated-query pass reports the encode-cache path.

The result keeps the item count, the host load, the OpenSearch segment count and the API git
sha so a table in docs/PERFORMANCE.md can be traced to one run.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from scripts.bench.retrieval_eval_metrics import latency_summary
from scripts.curation._project_worker_utils import curation_api_prefix
from src.domain.embedding_spaces import LEGACY_ITEM_VECTOR_FIELD
from src.domain.embedding_state import embedded_clause


if TYPE_CHECKING:
    import argparse


BAND_TARGETS: tuple[tuple[str, float], ...] = (
    ('rare', 0.005),
    ('about_3pct', 0.03),
    ('about_15pct', 0.15),
    ('about_45pct', 0.45),
)
CLASS_FIELD = 'detector_class_name'


def parse_server_timing(header: str) -> dict[str, float]:
    """``name;dur=1.23, name2;dur=4`` to ``{name: 1.23, name2: 4.0}`` (milliseconds)."""
    out: dict[str, float] = {}
    for part in header.split(','):
        name, _, rest = part.strip().partition(';dur=')
        if name and rest:
            out[name] = float(rest)
    return out


def pick_bands(
    counts: dict[str, int], total: int, targets: tuple[tuple[str, float], ...] = BAND_TARGETS
) -> list[dict[str, Any]]:
    """For each target share, the class whose measured share is nearest (each class once)."""
    if total <= 0:
        raise ValueError('no embedded items to take selectivity bands from')
    used: set[str] = set()
    bands: list[dict[str, Any]] = []
    for label, target in targets:
        candidates = [(abs(n / total - target), name, n) for name, n in counts.items()]
        for _, name, n in sorted(candidates):
            if name in used:
                continue
            used.add(name)
            bands.append({'band': label, 'class': name, 'items': n, 'share': n / total})
            break
    return bands


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-stage percentiles of ``stages`` plus the client wall and the served total."""
    names = sorted({k for r in rows for k in r['stages']})
    out: dict[str, Any] = {
        'client_wall_ms': latency_summary([r['wall_ms'] for r in rows]),
        'served_total_ms': latency_summary([r['total_ms'] for r in rows]),
    }
    for name in names:
        out[name] = latency_summary([r['stages'][name] for r in rows if name in r['stages']])
    return out


def one_request(
    client: httpx.Client, url: str, query: str, extra: dict[str, str], page_size: int
) -> dict[str, Any]:
    t0 = time.perf_counter()
    r = client.get(url, params={'q': query, 'page_size': page_size, **extra})
    wall = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    return {
        'wall_ms': wall,
        'total_ms': float(r.json().get('total_time_ms', 0.0)),
        'stages': parse_server_timing(r.headers.get('Server-Timing', '')),
    }


def load_captions(path: Path, n: int, seed: int) -> list[str]:
    caps = json.loads(path.read_text())['annotations']
    texts = sorted({c['caption'].strip().rstrip('.') for c in caps if c['caption'].strip()})
    random.Random(seed).shuffle(texts)
    return texts[:n]


def opensearch_facts(os_url: str, index: str) -> dict[str, Any]:
    """Item count, class counts of embedded items, segment count and heap, read-only."""
    with httpx.Client(base_url=os_url, timeout=120) as http:
        agg = http.post(
            f'/{index}/_search',
            json={
                'size': 0,
                'track_total_hits': True,
                'query': embedded_clause(LEGACY_ITEM_VECTOR_FIELD),
                'aggs': {'c': {'terms': {'field': CLASS_FIELD, 'size': 200}}},
            },
        ).json()
        segs = http.get(f'/_cat/segments/{index}', params={'format': 'json'}).json()
        heap = http.get('/_nodes/stats/jvm', params={'filter_path': '**.heap_max_in_bytes'}).json()
    buckets = agg['aggregations']['c']['buckets']
    return {
        'embedded_items': agg['hits']['total']['value'],
        'class_counts': {b['key']: b['doc_count'] for b in buckets},
        'segments': len(segs),
        'heap_max_bytes': [
            n['jvm']['mem']['heap_max_in_bytes'] for n in (heap.get('nodes') or {}).values()
        ],
    }


def run_condition(
    client: httpx.Client, url: str, queries: list[str], extra: dict[str, str], a: argparse.Namespace
) -> dict[str, Any]:
    warm, measured = queries[: a.warmup], queries[a.warmup : a.warmup + a.measured]
    cold = [one_request(client, url, q, extra, a.page_size) for q in warm]
    rows = [one_request(client, url, q, extra, a.page_size) for q in measured]
    slow = [r for r in rows if r['total_ms'] > a.slow_ms]
    return {
        'filter': extra,
        'cold': summarize(cold),
        'measured': summarize(rows),
        'slow_requests': slow[:20],
        'n_slow': len(slow),
    }


def run(a: argparse.Namespace) -> dict[str, Any]:
    url = f'{a.api}{curation_api_prefix()}/projects/{a.project}/search/text'
    facts = opensearch_facts(a.opensearch_url, a.index)
    bands = pick_bands(facts['class_counts'], facts['embedded_items'])
    need = (a.warmup + a.measured) * (1 + len(bands)) + a.repeat
    queries = load_captions(Path(a.captions_file), need, a.seed)
    if len(queries) < need:
        raise SystemExit(f'only {len(queries)} distinct captions, need {need}')
    load0 = os.getloadavg()
    conditions: dict[str, Any] = {}
    with httpx.Client(timeout=120.0) as client:
        offset = 0
        step = a.warmup + a.measured
        conditions['unfiltered'] = run_condition(
            client, url, queries[offset : offset + step], {}, a
        )
        for band in bands:
            offset += step
            extra = {CLASS_FIELD: band['class']}
            res = run_condition(client, url, queries[offset : offset + step], extra, a)
            conditions[band['band']] = {**res, 'band_facts': band}
        # the encode-cache path: one query repeated, the first request primes the cache
        repeated = [one_request(client, url, queries[-1], {}, a.page_size) for _ in range(a.repeat)]
        conditions['repeated_query'] = summarize(repeated[1:])
    return {
        'schema_version': 1,
        'api': a.api,
        'project': a.project,
        'opensearch': {k: v for k, v in facts.items() if k != 'class_counts'},
        'captions': {'source': Path(a.captions_file).name, 'seed': a.seed},
        'params': {
            'warmup': a.warmup,
            'n': a.measured,
            'page_size': a.page_size,
            'slow_ms': a.slow_ms,
        },
        'host_load_start': load0,
        'host_load_end': os.getloadavg(),
        'conditions': conditions,
        'note': a.note,
    }


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument('--opensearch-url', default='http://127.0.0.1:4627')
    p.add_argument('--index', default='', help='the project items index (read only)')
    p.add_argument('--captions-file', default='', help='COCO captions_*2017.json')
    p.add_argument('--warmup', type=int, default=128, help='discarded; keep above the worker count')
    p.add_argument('--measured', type=int, default=300, help='measured requests per condition')
    p.add_argument('--repeat', type=int, default=100)
    p.add_argument('--page-size', type=int, default=20)
    p.add_argument('--slow-ms', type=float, default=300.0)
    p.add_argument('--note', default='')
