"""Runs for retrieval_eval.py: ingest helpers and the four evaluations."""

from __future__ import annotations

import itertools
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from scripts.bench import retrieval_eval_data as data
from scripts.bench.retrieval_eval_metrics import jaccard, latency_summary, rank_of, with_ci
from scripts.curation._project_worker_utils import curation_api_prefix, scoped_url


if TYPE_CHECKING:
    import argparse

MIN_JACCARD = 0.5
CURATION_MAX_PAGES = 5
LOAD_SAMPLE_EVERY = 100


def load_sample(work: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = json.loads((Path(work) / 'sample.json').read_text())['rows']
    return rows


def _load1() -> float:
    import os

    return os.getloadavg()[0]


def core_id(coco_id: int) -> str:
    return f'rq{coco_id}'


def curation_base(a: argparse.Namespace) -> str:
    return scoped_url(a.api, curation_api_prefix(), a.project, '')


def ingest_curation(
    api: str, project: str, work: Path, sample: list[dict[str, Any]], batch: int
) -> dict[str, int]:
    """Upload through the product; returns {curation image_id: coco_id}."""
    mapping: dict[str, int] = {}
    with httpx.Client(timeout=600.0) as client:
        for i in range(0, len(sample), batch):
            rows = sample[i : i + batch]
            files = [
                (
                    'images',
                    (r['file_name'], (work / 'images' / r['file_name']).read_bytes(), 'image/jpeg'),
                )
                for r in rows
            ]
            form = {
                'image_paths': json.dumps([r['file_name'] for r in rows]),
                'source': 'rq_retrieval',
            }
            resp = client.post(
                scoped_url(api, curation_api_prefix(), project, '/ingest/upload'),
                files=files,
                data=form,
            )
            resp.raise_for_status()
            by_name = {r['file_name']: r['coco_id'] for r in rows}
            for res in resp.json()['results']:
                if res.get('status') in ('success', 'duplicate'):
                    mapping[res['image_id']] = by_name[res['source_identifier']]
                else:
                    print('ingest problem', res.get('source_identifier'), res.get('error'))
            print(f'{i + len(rows)}/{len(sample)}', flush=True)
    return mapping


def _core_ingest_one(
    api: str, path: Path, image_id: str, near_duplicate: bool, threshold: float
) -> dict[str, Any]:
    with httpx.Client(timeout=300.0) as client:
        resp = client.post(
            f'{api}/ingest',
            files={'image': (path.name, path.read_bytes(), 'image/jpeg')},
            data={
                'image_id': image_id,
                'image_path': path.name,
                'detect_near_duplicates': str(near_duplicate).lower(),
                'near_duplicate_threshold': str(threshold),
            },
        )
    body: dict[str, Any] = resp.json() if resp.content else {}
    return {'http': resp.status_code, 'image_id': image_id, **body}


def ingest_core(api: str, work: Path, sample: list[dict[str, Any]], workers: int) -> dict[str, Any]:
    jobs = [(work / 'images' / r['file_name'], core_id(r['coco_id'])) for r in sample]
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(lambda j: _core_ingest_one(api, j[0], j[1], True, 0.99), jobs))
    flagged = [r for r in results if r.get('near_duplicate')]
    out = {
        'n': len(results),
        'http': {
            str(k): sum(1 for r in results if r['http'] == k) for k in {r['http'] for r in results}
        },
        'status': {
            str(k): sum(1 for r in results if r.get('status') == k)
            for k in {r.get('status') for r in results}
        },
        'near_duplicate_flagged_among_distinct_images': len(flagged),
        'flagged_examples': [
            {'image_id': r['image_id'], **r['near_duplicate']} for r in flagged[:20]
        ],
    }
    (work / 'core_ingest.json').write_text(json.dumps(out, indent=1))
    return out


def core_cleanup(api: str, sample: list[dict[str, Any]], include_variants: bool) -> dict[str, Any]:
    ids = [core_id(r['coco_id']) for r in sample]
    if include_variants:
        ids += [
            f'rqv_{r["coco_id"]}_{k}_{t}'
            for r in sample
            for k in data.VARIANTS
            for t in ('99', '95', '90')
        ]
    deleted = failed = 0
    with httpx.Client(timeout=60.0) as client:
        for iid in ids:
            resp = client.delete(f'{api}/query/image/{iid}')
            if resp.status_code == 200 and resp.json().get('deleted', resp.json().get('status')):
                deleted += 1
            else:
                failed += 1
    return {'attempted': len(ids), 'ok_responses': deleted, 'other': failed}


def build_queries(sample: list[dict[str, Any]], captions: int, limit: int) -> list[tuple[int, str]]:
    qs = [(r['coco_id'], c) for r in sample for c in r['captions'][:captions]]
    return qs[:limit] if limit else qs


def _summ(ranks: list[int | None], lat: list[float], extra: dict[str, Any]) -> dict[str, Any]:
    return {'quality': with_ci(ranks), 'latency_ms': latency_summary(lat), **extra}


def _failures(
    rows: list[dict[str, Any]], cap: dict[int, str], limit: int = 12
) -> list[dict[str, Any]]:
    bad = [r for r in rows if r['rank'] is None or r['rank'] > 10][:limit]
    return [{**r, 'top1_caption': cap.get(r['top1'])} for r in bad]


def eval_text_curation(
    client: httpx.Client, a: argparse.Namespace, sample: list[dict[str, Any]]
) -> dict[str, Any]:
    mapping: dict[str, int] = json.loads(
        (Path(a.work) / f'curation_map_{a.project}.json').read_text()
    )
    cap = {r['coco_id']: r['captions'][0] for r in sample}
    queries = build_queries(sample, a.captions, a.limit)
    ranks: list[int | None] = []
    crop_ranks: list[int | None] = []
    lat: list[float] = []
    server: list[float] = []
    loads: list[float] = []
    rows: list[dict[str, Any]] = []
    distinct_in_page: list[int] = []
    extra_pages = 0
    for n, (gt, text) in enumerate(queries):
        if n % LOAD_SAMPLE_EVERY == 0:
            loads.append(_load1())
        seen: list[int] = []
        crop_pos = None
        pos = 0
        top_score = gt_score = None
        for page in range(1, CURATION_MAX_PAGES + 1):
            t0 = time.perf_counter()
            r = client.get(
                f'{curation_base(a)}/search/text',
                params={'q': text, 'page': page, 'page_size': a.depth},
            )
            dt = (time.perf_counter() - t0) * 1000
            r.raise_for_status()
            body = r.json()
            if page == 1:
                lat.append(dt)
                server.append(float(body.get('total_time_ms', 0.0)))
            else:
                extra_pages += 1
            for it in body['items']:
                pos += 1
                coco = mapping.get(it['image_id'])
                if top_score is None:
                    top_score = it['semantic_score']
                if coco == gt and crop_pos is None:
                    crop_pos = pos
                    gt_score = it['semantic_score']
                if coco is not None and coco not in seen:
                    seen.append(coco)
            if page == 1:
                distinct_in_page.append(len(seen))
            if len(body['items']) < a.depth or (crop_pos is not None and len(seen) >= 10):
                break
        rank = rank_of([str(s) for s in seen], {str(gt)})
        ranks.append(rank)
        crop_ranks.append(crop_pos)
        rows.append(
            {
                'gt': gt,
                'query': text,
                'rank': rank,
                'top1': seen[0] if seen else None,
                'top_score': top_score,
                'gt_score': gt_score,
            }
        )
    return {
        'route': 'GET /curation/projects/{project}/search/text',
        'params': {
            'captions': a.captions,
            'n_queries': len(queries),
            'page_size': a.depth,
            'ranking': 'distinct images by first appearance of any of their crops',
        },
        'load1_samples': loads,
        **_summ(
            ranks,
            lat,
            {
                'server_total_time_ms': latency_summary(server),
                'crop_level_rank_quality': with_ci(crop_ranks),
                'distinct_images_in_first_page_mean': sum(distinct_in_page)
                / max(1, len(distinct_in_page)),
                'extra_pages_fetched': extra_pages,
                'failure_examples': _failures(rows, cap),
            },
        ),
    }


def eval_text_core(
    client: httpx.Client, a: argparse.Namespace, sample: list[dict[str, Any]]
) -> dict[str, Any]:
    cap = {r['coco_id']: r['captions'][0] for r in sample}
    queries = build_queries(sample, a.captions, a.limit)
    ranks: list[int | None] = []
    lat: list[float] = []
    server: list[float] = []
    loads: list[float] = []
    rows: list[dict[str, Any]] = []
    empty = 0
    for n, (gt, text) in enumerate(queries):
        if n % LOAD_SAMPLE_EVERY == 0:
            loads.append(_load1())
        t0 = time.perf_counter()
        r = client.post(
            f'{a.api}/search/text', params={'text': text, 'top_k': 100, 'min_score': a.min_score}
        )
        lat.append((time.perf_counter() - t0) * 1000)
        r.raise_for_status()
        body = r.json()
        server.append(float(body.get('search_time_ms', 0.0)))
        ids = [x['image_id'] for x in body['results']]
        empty += not ids
        rank = rank_of(ids, {core_id(gt)})
        ranks.append(rank)
        top1 = int(ids[0][2:]) if ids and ids[0].startswith('rq') and ids[0][2:].isdigit() else None
        rows.append(
            {
                'gt': gt,
                'query': text,
                'rank': rank,
                'top1': top1,
                'top_score': body['results'][0]['score'] if ids else None,
            }
        )
    return {
        'route': 'POST /search/text (MobileCLIP, visual_search_global)',
        'params': {
            'captions': a.captions,
            'n_queries': len(queries),
            'top_k': 100,
            'min_score': a.min_score,
        },
        'load1_samples': loads,
        **_summ(
            ranks,
            lat,
            {
                'server_search_time_ms': latency_summary(server),
                'empty_result_queries': empty,
                'failure_examples': _failures(rows, cap),
            },
        ),
    }


def eval_image_core(
    client: httpx.Client, a: argparse.Namespace, sample: list[dict[str, Any]]
) -> dict[str, Any]:
    cats = {r['coco_id']: set(r['categories']) for r in sample}
    rel_count = {
        q: sum(1 for o, c in cats.items() if o != q and jaccard(cats[q], c) >= MIN_JACCARD)
        for q in cats
    }
    queries = [r for r in sample if cats[r['coco_id']]]
    if a.limit:
        queries = queries[: a.limit]
    prec: list[float] = []
    prec_norm: list[float] = []
    self_rank1 = 0
    lat: list[float] = []
    loads: list[float] = []
    fails: list[dict[str, Any]] = []
    for n, r in enumerate(queries):
        if n % LOAD_SAMPLE_EVERY == 0:
            loads.append(_load1())
        q = r['coco_id']
        t0 = time.perf_counter()
        resp = client.post(
            f'{a.api}/search/image',
            params={'top_k': 11, 'min_score': 0.0},
            files={
                'image': (
                    r['file_name'],
                    (Path(a.work) / 'images' / r['file_name']).read_bytes(),
                    'image/jpeg',
                )
            },
        )
        lat.append((time.perf_counter() - t0) * 1000)
        resp.raise_for_status()
        ids = [x['image_id'] for x in resp.json()['results']]
        self_rank1 += bool(ids) and ids[0] == core_id(q)
        others = [int(i[2:]) for i in ids if i != core_id(q) and i[2:].isdigit()][:10]
        hits = sum(1 for o in others if jaccard(cats[q], cats[o]) >= MIN_JACCARD)
        prec.append(hits / 10)
        ceiling = min(10, rel_count[q]) / 10
        if ceiling > 0:
            prec_norm.append(hits / 10 / ceiling)
        if hits == 0 and ceiling > 0 and len(fails) < 12:
            fails.append(
                {
                    'query': q,
                    'categories': sorted(cats[q]),
                    'top3': [{'coco_id': o, 'categories': sorted(cats[o])} for o in others[:3]],
                }
            )
    n_ = max(1, len(prec))
    return {
        'route': 'POST /search/image (MobileCLIP, visual_search_global); leave-one-out, self hit removed',
        'params': {
            'n_queries': len(queries),
            'relevance': f'category-set Jaccard >= {MIN_JACCARD} (proxy)',
            'pool': len(sample),
        },
        'load1_samples': loads,
        'precision@10': {
            'mean': sum(prec) / n_,
            'zero_precision_queries': sum(1 for p in prec if p == 0),
        },
        'precision@10_over_ceiling_mean': sum(prec_norm) / max(1, len(prec_norm)),
        'chance_precision@10': sum(rel_count[r['coco_id']] for r in queries)
        / n_
        / max(1, len(sample) - 1),
        'mean_relevant_in_pool': sum(rel_count[r['coco_id']] for r in queries) / n_,
        'queries_with_no_relevant_in_pool': sum(1 for r in queries if rel_count[r['coco_id']] == 0),
        'self_hit_at_rank1': self_rank1,
        'latency_ms': latency_summary(lat),
        'failure_examples': fails,
    }


FILTER_CLASSES = ('person', 'car', 'chair', 'dog', 'cup', 'bottle', 'bus', 'clock', 'cat', 'bowl')
FILTER_REPS = 30
PAGE_CHECKS = 20


def _search(
    client: httpx.Client, a: argparse.Namespace, params: dict[str, Any]
) -> tuple[httpx.Response, float]:
    t0 = time.perf_counter()
    r = client.get(f'{curation_base(a)}/search/text', params=params)
    return r, (time.perf_counter() - t0) * 1000


def _class_ok(item: dict[str, Any], cls: str) -> bool:
    return cls in (
        item.get('class_name'),
        item.get('detector_class_name'),
        item.get('proposal_name'),
    )


def _scores_sorted(items: list[dict[str, Any]]) -> bool:
    sc = [i['semantic_score'] for i in items]
    return all(x >= y - 1e-9 for x, y in itertools.pairwise(sc))


def eval_filters(
    client: httpx.Client, a: argparse.Namespace, sample: list[dict[str, Any]]
) -> dict[str, Any]:
    out: dict[str, Any] = {
        'route': 'GET /curation/projects/{project}/search/text with class_name / cluster_id'
    }
    violations: list[dict[str, Any]] = []
    per_class: list[dict[str, Any]] = []
    loads = [_load1()]
    for cls in FILTER_CLASSES:
        texts = [c for r in sample if cls in r['categories'] for c in r['captions'][:1]][
            :FILTER_REPS
        ]
        lat_f: list[float] = []
        lat_u: list[float] = []
        agree: list[float] = []
        bad = unsorted = empty = covered = 0
        n_hits: list[int] = []
        for t in texts:
            rf, df = _search(client, a, {'q': t, 'class_name': cls, 'page_size': 10})
            ru, du = _search(client, a, {'q': t, 'page_size': 10})
            rf.raise_for_status()
            ru.raise_for_status()
            lat_f.append(df)
            lat_u.append(du)
            items = rf.json()['items']
            n_hits.append(len(items))
            empty += not items
            wrong = [i for i in items if not _class_ok(i, cls)]
            bad += len(wrong)
            unsorted += not _scores_sorted(items)
            if wrong and len(violations) < 10:
                violations.append(
                    {
                        'filter': f'class_name={cls}',
                        'query': t,
                        'item_class': wrong[0].get('class_name'),
                        'detector': wrong[0].get('detector_class_name'),
                    }
                )
            deep, _ = _search(client, a, {'q': t, 'page_size': 200})
            deep_items = [i for i in deep.json()['items'] if _class_ok(i, cls)][:10]
            if len(deep_items) == 10:
                covered += 1
                agree.append(len({i['id'] for i in deep_items} & {i['id'] for i in items}) / 10)
        per_class.append(
            {
                'class_name': cls,
                'queries': len(texts),
                'filtered_latency_ms': latency_summary(lat_f),
                'unfiltered_latency_ms': latency_summary(lat_u),
                'items_returned_mean': sum(n_hits) / max(1, len(n_hits)),
                'empty_result_queries': empty,
                'items_violating_filter': bad,
                'score_order_violations': unsorted,
                'top10_agreement_with_filtered_unfiltered_deep_list': {
                    'queries_with_comparable_deep_list': covered,
                    'mean_overlap@10': sum(agree) / len(agree) if agree else None,
                },
            }
        )
    out['class_name'] = per_class
    loads.append(_load1())

    # cluster_id: take the clusters seen in a broad search, then filter by each.
    seen: dict[int, int] = {}
    for r in sample[:60]:
        resp, _ = _search(client, a, {'q': r['captions'][0], 'page_size': 200})
        for it in resp.json()['items']:
            if it.get('cluster_id') is not None:
                seen[it['cluster_id']] = seen.get(it['cluster_id'], 0) + 1
    clusters = [c for c, _ in sorted(seen.items(), key=lambda kv: -kv[1])[:10]]
    per_cluster = []
    for cid in clusters:
        lat: list[float] = []
        bad = unsorted = empty = 0
        for r in sample[:FILTER_REPS]:
            resp, dt = _search(
                client, a, {'q': r['captions'][0], 'cluster_id': cid, 'page_size': 10}
            )
            resp.raise_for_status()
            lat.append(dt)
            items = resp.json()['items']
            empty += not items
            wrong = [i for i in items if i.get('cluster_id') != cid]
            bad += len(wrong)
            unsorted += not _scores_sorted(items)
            if wrong and len(violations) < 10:
                violations.append(
                    {'filter': f'cluster_id={cid}', 'got_cluster': wrong[0].get('cluster_id')}
                )
        per_cluster.append(
            {
                'cluster_id': cid,
                'latency_ms': latency_summary(lat),
                'empty_result_queries': empty,
                'items_violating_filter': bad,
                'score_order_violations': unsorted,
            }
        )
    out['cluster_id'] = per_cluster
    loads.append(_load1())

    # pagination: pages of 20 must concatenate to one page of 60 with no repeats; scores must not rise.
    page_issues = {
        'queries': 0,
        'overlap_between_pages': 0,
        'differs_from_single_page': 0,
        'score_rise_across_pages': 0,
        'total_equals_page_size_times_page': 0,
    }
    for r in sample[:PAGE_CHECKS]:
        q = r['captions'][0]
        pages = []
        totals = []
        for pg in (1, 2, 3):
            resp, _ = _search(client, a, {'q': q, 'page': pg, 'page_size': 20})
            resp.raise_for_status()
            pages.append(resp.json()['items'])
            totals.append(resp.json()['total'])
        one, _ = _search(client, a, {'q': q, 'page': 1, 'page_size': 60})
        flat = [i['id'] for p in pages for i in p]
        page_issues['queries'] += 1
        page_issues['overlap_between_pages'] += len(flat) != len(set(flat))
        page_issues['differs_from_single_page'] += flat != [i['id'] for i in one.json()['items']]
        allsc = [i['semantic_score'] for p in pages for i in p]
        page_issues['score_rise_across_pages'] += any(
            y > x + 1e-9 for x, y in itertools.pairwise(allsc)
        )
        page_issues['total_equals_page_size_times_page'] += totals == [20, 40, 60]
    out['pagination'] = page_issues

    # min_score and request validation
    ms = _search(client, a, {'q': sample[0]['captions'][0], 'min_score': 0.3, 'page_size': 200})[
        0
    ].json()['items']
    out['min_score_0.3'] = {
        'returned': len(ms),
        'below_threshold': sum(1 for i in ms if i['semantic_score'] < 0.3),
    }
    codes = {}
    probes: dict[str, dict[str, Any]] = {
        'page_size_201': {'q': 'dog', 'page_size': 201},
        'empty_q': {'q': ''},
        'page_0': {'q': 'dog', 'page': 0},
        'class_name_unknown': {'q': 'dog', 'class_name': 'zz_no_such_class'},
        'cluster_id_unknown': {'q': 'dog', 'cluster_id': 987654321},
        'min_score_1.5': {'q': 'dog', 'min_score': 1.5},
        'page_beyond_k_cap': {'q': 'dog', 'page': 50, 'page_size': 200},
    }
    for label, prm in probes.items():
        resp, _ = _search(client, a, prm)
        body = resp.json() if resp.content else {}
        codes[label] = {
            'http': resp.status_code,
            'items': len(body.get('items', [])) if resp.status_code == 200 else None,
            'detail': str(body.get('detail'))[:120] if resp.status_code != 200 else None,
        }
    out['request_validation'] = codes
    out['filter_violations_examples'] = violations
    out['load1_samples'] = loads
    return out


def _poll_dup(client: httpx.Client, api: str) -> dict[str, Any]:
    stats: dict[str, Any] = client.get(f'{api}/query/duplicates/stats').json()
    return stats


def eval_dedupe(
    client: httpx.Client, a: argparse.Namespace, sample: list[dict[str, Any]]
) -> dict[str, Any]:
    """Near-duplicate variants through core ingest. Set A/B/C of --variants images at thresholds 0.99/0.95/0.90."""
    work = Path(a.work)
    vdir = work / 'variants'
    vdir.mkdir(exist_ok=True)
    out: dict[str, Any] = {
        'route': 'POST /ingest (near-duplicate grouping), GET /query/duplicates*',
        'stats_before': _poll_dup(client, a.api),
    }
    sets = {
        '99': (0.99, sample[0 : a.variants]),
        '95': (0.95, sample[a.variants : 2 * a.variants]),
        '90': (0.90, sample[2 * a.variants : 3 * a.variants]),
    }
    per: dict[str, Any] = {}
    jobs = []
    for tag, (thr, rows) in sets.items():
        for r in rows:
            for kind in data.VARIANTS:
                path = vdir / f'{r["coco_id"]}_{kind}_{tag}.jpg'
                path.write_bytes(data.make_variant(work / 'images' / r['file_name'], kind))
                jobs.append((tag, thr, kind, r['coco_id'], path))
    with ThreadPoolExecutor(2) as pool:
        results = list(
            pool.map(
                lambda j: _core_ingest_one(a.api, j[4], f'rqv_{j[3]}_{j[2]}_{j[0]}', True, j[1]),
                jobs,
            )
        )
    group_cache: dict[str, set[str]] = {}

    def members(gid: str) -> set[str]:
        if gid not in group_cache:
            body = client.get(f'{a.api}/query/duplicates/{gid}').json()
            group_cache[gid] = {m['image_id'] for m in body.get('members', [])}
        return group_cache[gid]

    for (tag, thr, kind, coco, _), res in zip(jobs, results, strict=True):
        key = f'thr_{tag}/{kind}'
        row = per.setdefault(
            key,
            {
                'threshold': thr,
                'kind': kind,
                'n': 0,
                'http_ok': 0,
                'grouped': 0,
                'grouped_with_original': 0,
                'similarities': [],
            },
        )
        row['n'] += 1
        row['http_ok'] += res['http'] in (200, 201)
        nd = res.get('near_duplicate')
        if nd:
            row['grouped'] += 1
            row['similarities'].append(nd['similarity'])
            row['grouped_with_original'] += core_id(coco) in members(nd['group_id'])
    for row in per.values():
        sims = sorted(row.pop('similarities'))
        row['similarity_min'] = sims[0] if sims else None
        row['similarity_median'] = sims[len(sims) // 2] if sims else None
    out['variants'] = per
    out['stats_after'] = _poll_dup(client, a.api)
    # exact re-upload of 20 originals
    exact = [
        _core_ingest_one(a.api, work / 'images' / r['file_name'], core_id(r['coco_id']), True, 0.99)
        for r in sample[:20]
    ]
    out['exact_reupload'] = {
        'n': 20,
        'status': {
            k: sum(1 for e in exact if e.get('status') == k)
            for k in {e.get('status') for e in exact}
        },
        'http': sorted({e['http'] for e in exact}),
    }
    lst = client.get(f'{a.api}/query/duplicates', params={'size': 100}).json()
    out['groups_listing_first_page'] = {
        'groups': len(lst.get('groups', [])),
        'total_groups': lst.get('total_groups'),
    }
    return out
