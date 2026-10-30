"""Probe for label operations on a seeded throw-away project.

Seed the project with ``cluster_list_probe.py seed`` first (synthetic items, no images or vectors).

    label_op_probe.py run  --api URL --opensearch URL --project SLUG --items 20000 --edits 50 \
        --container NAME --out run.json
    label_op_probe.py run  ... --label-only          # time the batch label on another build, then revert it
    label_op_probe.py ops  --api URL --opensearch URL --project SLUG --count 200

``run`` picks N items with a seeded random filter selection, snapshots the project's class state
and ``class_name`` histogram straight from OpenSearch, times the batch label, relabels K of the
written items one by one, times the undo dry run and the undo (sampling the API process RSS every
500 ms), and audits every item of the project against the snapshot: an item that should be
restored and is not is a lost write, any other item that changed is an extra write. The K edited
items must keep their edit. ``ops`` writes many small operations so the list route can be timed
(``cluster_list_probe.py probe --endpoint label_ops``). Only the named project is touched.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
from cluster_list_probe import COCO, NOISY_LOAD, PREFIX, _load1


STATE_FIELDS = (
    'class_id',
    'class_name',
    'class_source',
    'label_source',
    'confidence',
    'class_detector',
    'class_detector_version',
    'class_labeler',
    'class_labeled_at',
    'class_validated',
    'cluster_id',
    'cluster_subid',
    'audit_outcome',
    'audit_human_name',
)
UNSET = '<unset>'
SCAN_PAGE = 5000
RSS_SAMPLE_S = 0.5
EDIT_CLASS = 'probe_b'
LABEL_CLASS = 'probe_a'


def state_of(source: dict[str, Any]) -> dict[str, Any]:
    state = {f: source.get(f) for f in STATE_FIELDS}
    state['class_validated'] = bool(state['class_validated'])
    return state


def _only_cluster_normalised(before: dict[str, Any], after: dict[str, Any]) -> bool:
    differing = {k for k in STATE_FIELDS if before.get(k) != after.get(k)}
    return (
        differing == {'cluster_id'}
        and bool(before.get('class_validated'))
        and before.get('class_id') is not None
        and after.get('cluster_id') == before.get('class_id')
    )


def audit_state(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
    *,
    edited: set[str],
    edit_state: dict[str, Any],
) -> dict[str, Any]:
    """Compare every item before and after. An edited item must carry the edit
    (``edit_state``'s fields); every other item must be exactly as before. Returns the
    items that differ: ``lost`` (written by the operation and not restored) and ``extra``
    (never written by it, changed anyway) are told apart by the caller through ``edited``."""
    wrong_edit = sorted(
        i for i in edited if any(after.get(i, {}).get(k) != v for k, v in edit_state.items())
    )
    differing = sorted(i for i in before if i not in edited and before[i] != after.get(i))
    # The restore puts a validated class back into its class cluster (cluster_id == class_id), so
    # an item the synthetic seed stored validated but in another cluster comes back normalised.
    normalised = [i for i in differing if _only_cluster_normalised(before[i], after.get(i, {}))]
    changed = [i for i in differing if i not in set(normalised)]
    missing = sorted(set(before) - set(after))
    return {
        'items': len(before),
        'edited': len(edited),
        'wrong_edit': len(wrong_edit),
        'normalised_cluster_only': len(normalised),
        'changed_or_not_restored': len(changed),
        'missing': len(missing),
        'examples': (changed + wrong_edit + missing)[:5],
        'diffs': {
            i: {
                k: [before[i].get(k), after.get(i, {}).get(k)]
                for k in STATE_FIELDS
                if before[i].get(k) != after.get(i, {}).get(k)
            }
            for i in changed[:5]
        },
    }


def expected_histogram(
    before: dict[str, int], edited_before: Counter[str], edit_name: str
) -> dict[str, int]:
    """The ``class_name`` histogram after the undo: the snapshot with the edited items
    moved to the edit's class."""
    out = Counter(before)
    out.subtract(edited_before)
    out[edit_name] += sum(edited_before.values())
    return {k: v for k, v in out.items() if v}


def rss_mb(pid: int) -> float | None:
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024
    except OSError:
        return None
    return None


class RssSampler:
    def __init__(self, pid: int | None) -> None:
        self.pid = pid
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while self.pid is not None and not self._stop.is_set():
            value = rss_mb(self.pid)
            if value is not None:
                self.samples.append(value)
            self._stop.wait(RSS_SAMPLE_S)

    def __enter__(self) -> RssSampler:
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        self._thread.join()


def container_pid(name: str | None) -> int | None:
    if not name:
        return None
    import subprocess  # nosec B404 - reads one container's pid

    out = subprocess.run(  # nosec B603 B607 - fixed argv, reads one container's pid
        ['docker', 'inspect', '-f', '{{.State.Pid}}', name],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout.strip()) or None


def scan_state(opensearch: str, index: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    cursor: list[Any] | None = None
    with httpx.Client(timeout=300) as client:
        while True:
            body: dict[str, Any] = {
                'size': SCAN_PAGE,
                '_source': list(STATE_FIELDS),
                'sort': [{'crop_id': 'asc'}],
                'query': {'match_all': {}},
            }
            if cursor is not None:
                body['search_after'] = cursor
            resp = client.post(f'{opensearch}/{index}/_search', json=body)
            resp.raise_for_status()
            hits = resp.json()['hits']['hits']
            if not hits:
                return out
            out.update({h['_id']: state_of(h.get('_source') or {}) for h in hits})
            cursor = hits[-1]['sort']


def histogram(opensearch: str, index: str) -> dict[str, int]:
    body = {
        'size': 0,
        'aggs': {'n': {'terms': {'field': 'class_name', 'size': 500, 'missing': UNSET}}},
    }
    resp = httpx.post(f'{opensearch}/{index}/_search', json=body, timeout=120)
    resp.raise_for_status()
    return {b['key']: b['doc_count'] for b in resp.json()['aggregations']['n']['buckets']}


def class_id(client: httpx.Client, base: str, name: str) -> int:
    listing = client.get(f'{base}/classes')
    listing.raise_for_status()
    for entry in listing.json().get('classes', listing.json().get('items', [])):
        if entry['class_name'] == name:
            return int(entry['class_id'])
    created = client.post(f'{base}/classes', json={'name': name})
    created.raise_for_status()
    return int(created.json()['class_id'])


def items_index(api: str, project: str) -> str:
    summary = httpx.get(f'{api}{PREFIX}/projects/{project}', timeout=60)
    summary.raise_for_status()
    return str(summary.json()['resources']['indexes']['items'])


def _undo_page(client: httpx.Client, base: str, ids: list[str]) -> None:
    """Undo one page; a project whose registry refresh failed under host load answers 409
    ``project_read_only`` for a moment, so that one answer is retried."""
    for attempt in range(6):
        resp = client.post(f'{base}/crops/label/undo_batch', json={'crop_ids': ids})
        if resp.status_code != 409 or 'project_read_only' not in resp.text:
            resp.raise_for_status()
            return
        time.sleep(2.0 * (attempt + 1))
    resp.raise_for_status()


def run(args: argparse.Namespace) -> dict[str, Any]:
    base = f'{args.api}{PREFIX}/projects/{args.project}'
    index = items_index(args.api, args.project)
    result: dict[str, Any] = {
        'project': args.project,
        'items_selected': args.items,
        'edits': args.edits,
        'seed': args.seed,
    }
    with httpx.Client(timeout=900) as client:
        t0 = time.perf_counter()
        client.get(f'{args.api}/health').raise_for_status()
        client.get(
            f'{base}/label_ops'
        )  # a cold project bootstraps its indexes on its first request
        result['health_ms'] = round((time.perf_counter() - t0) * 1000, 1)
        label_id = class_id(client, base, LABEL_CLASS)
        edit_id = class_id(client, base, EDIT_CLASS)
        before = scan_state(args.opensearch, index)
        hist_before = histogram(args.opensearch, index)
        selection = {'filter': {}, 'limit': args.items, 'sample': 'random', 'seed': args.seed}
        load_start = _load1()

        t0 = time.perf_counter()
        resp = client.put(
            f'{base}/crops/batch_label', json={'selection': selection, 'class_id': label_id}
        )
        label_s = time.perf_counter() - t0
        resp.raise_for_status()
        body = resp.json()
        written = body['updated_ids']
        result.update(label_s=round(label_s, 2), updated=body['updated'], op_id=body.get('op_id'))

        if (
            args.label_only
        ):  # revert on a build without operations, in the old route's 5,000-id pages
            result['load1'] = [load_start, _load1()]
            result['noisy'] = max(result['load1']) > NOISY_LOAD
            if args.out:
                Path(args.out).write_text(
                    json.dumps(result, indent=2)
                )  # the timing, before the revert
            for start in range(0, len(written), 5000):
                _undo_page(client, base, written[start : start + 5000])
            return result

        edited = written[: args.edits]
        edited_before = Counter(before[i]['class_name'] or UNSET for i in edited)
        for crop_id in edited:
            client.put(
                f'{base}/crops/{crop_id}/label', json={'class_id': edit_id}
            ).raise_for_status()

        op_url = f'{base}/label_ops/{body["op_id"]}/undo'
        t0 = time.perf_counter()
        dry = client.post(op_url, json={'dry_run': True})
        result['dry_run_s'] = round(time.perf_counter() - t0, 2)
        dry.raise_for_status()
        result['dry_run'] = dry.json()

        pid = container_pid(args.container)
        baseline = rss_mb(pid) if pid else None
        with RssSampler(pid) as sampler:
            t0 = time.perf_counter()
            undo = client.post(op_url, json={'dry_run': False})
            result['undo_s'] = round(time.perf_counter() - t0, 2)
        undo.raise_for_status()
        result['undo'] = undo.json()
        result['rss_baseline_mb'] = None if baseline is None else round(baseline, 1)
        result['rss_peak_mb'] = round(max(sampler.samples), 1) if sampler.samples else None
        result['rss_peak_delta_mb'] = (
            None
            if baseline is None or not sampler.samples
            else round(max(sampler.samples) - baseline, 1)
        )
        result['rss_samples'] = len(sampler.samples)
        result['load1'] = [load_start, _load1()]
        result['noisy'] = max(result['load1']) > NOISY_LOAD

        # Not a dict literal: the single-writer gate scans those for a validated class.
        edit_state = {'class_id': edit_id, 'class_name': EDIT_CLASS}
        edit_state['class_validated'] = True
        after = scan_state(args.opensearch, index)
        result['audit'] = audit_state(
            before,
            after,
            edited=set(edited),
            edit_state=edit_state,
        )
        hist_after = histogram(args.opensearch, index)
        result['histogram_matches'] = hist_after == expected_histogram(
            hist_before, edited_before, EDIT_CLASS
        )
        tagged = httpx.post(
            f'{args.opensearch}/{index}/_count',
            json={'query': {'exists': {'field': 'label_op_id'}}},
            timeout=120,
        ).json()['count']
        result['still_tagged'] = tagged
        und = result['undo']
        result['counts_ok'] = (
            und['restored'] == args.items - args.edits
            and und['kept_changed'] == args.edits
            and und['conflicts'] == 0
            and und['matched']
            == sum(
                und[k]
                for k in (
                    'restored',
                    'already_undone',
                    'kept_changed',
                    'kept_holdout',
                    'kept_stale_class',
                    'conflicts',
                )
            )
        )
    return result


def register_classes(args: argparse.Namespace) -> None:
    """Give the project the registry the synthetic items assume: the COCO names in order, so a
    class id on an item names the same class in the registry (an item whose class the registry
    no longer names that way is, correctly, not restored), then the two classes the probe writes."""
    base = f'{args.api}{PREFIX}/projects/{args.project}'
    with httpx.Client(timeout=300) as client:
        for name in COCO:
            class_id(client, base, name)
        class_id(client, base, LABEL_CLASS)
        class_id(client, base, EDIT_CLASS)
    print(f'registered {len(COCO) + 2} classes')


def many_ops(args: argparse.Namespace) -> None:
    base = f'{args.api}{PREFIX}/projects/{args.project}'
    index = items_index(args.api, args.project)
    ids = list(scan_state(args.opensearch, index))[: args.count * 10]
    with httpx.Client(timeout=300) as client:
        label_id = class_id(client, base, LABEL_CLASS)
        for n in range(args.count):
            chunk = ids[n * 10 : n * 10 + 10]
            client.put(
                f'{base}/crops/batch_label', json={'crop_ids': chunk, 'class_id': label_id}
            ).raise_for_status()
    print(f'wrote {args.count} operations')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='cmd', required=True)
    for name in ('run', 'ops', 'classes'):
        p = sub.add_parser(name)
        p.add_argument('--api', required=True)
        p.add_argument('--opensearch', required=True)
        p.add_argument('--project', required=True)
    runp = sub.choices['run']
    runp.add_argument('--items', type=int, default=10_000)
    runp.add_argument('--edits', type=int, default=50)
    runp.add_argument('--seed', type=int, default=1)
    runp.add_argument('--container', help='docker container whose process RSS is sampled')
    runp.add_argument('--label-only', action='store_true')
    runp.add_argument('--out')
    sub.choices['ops'].add_argument('--count', type=int, default=200)
    args = parser.parse_args()
    if args.cmd == 'ops':
        many_ops(args)
        return
    if args.cmd == 'classes':
        register_classes(args)
        return
    result = run(args)
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(text)
    print(text)


if __name__ == '__main__':
    main()
