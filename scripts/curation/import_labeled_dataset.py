#!/usr/bin/env python3
"""Import an already-labeled dataset (YOLO, COCO or an OpenProcessor export)
through ``POST {api_base}/projects/{project}/datasets/imports``.

A thin client of the import API: the SERVER scans the dataset, maps its
classes by name, imports the labels and images, and keeps the ledger, resume
and undo. This script previews, builds the class mapping from the flags,
starts (or resumes) the import and polls it to the end.

``--dataset`` is a path on the SERVER (a mounted source root). ``--path-map
LOCAL=SERVER`` rewrites a local path prefix to it.

Class mapping (every dataset class with boxes needs one)::

    --accept-suggestions         take exact / case-insensitive / region matches
    --map CLASS=CLASS_ID         map to an existing class
    --create CLASS[=NAME]        create a class (default name: the dataset's)
    --skip CLASS                 drop its boxes (the frames stay unlabeled)
    --region CLASS               a region class (needs an active region profile)
    --images-only                skip EVERY class: index the images, no labels

``--state-dir DIR`` writes ``DIR/ingested/<split>.jsonl`` (one line per image:
``image``, ``server_path``, ``image_id``, ``status``, ``positive``), the
cohort ``eval_regions_vs_gt.py --state-dir`` reads.

Usage::

    python3 scripts/curation/import_labeled_dataset.py --dataset /data/source/ds \\
        --accept-suggestions --dry-run
    python3 scripts/curation/import_labeled_dataset.py --project cars \\
        --dataset /data/source/ds --map Car=2 --create automobile=car --skip person
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import httpx


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from src.config.curation import base_curation_config as get_curation_config
from src.services.projects.script_binding import add_project_argument, bind_script_project


logger = logging.getLogger('import_labeled_dataset')

_TERMINAL = frozenset(
    {'completed', 'completed_with_errors', 'failed', 'cancelled', 'interrupted', 'undone'}
)


class ImportCliError(Exception):
    pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    add_project_argument(p)
    p.add_argument(
        '--api-base',
        default=f'http://localhost:4603{get_curation_config().api_prefix}',
        help='API base including the curation prefix (default: %(default)s)',
    )
    p.add_argument('--dataset', required=True, help='Dataset path on the server')
    p.add_argument('--path-map', default=None, help='LOCAL=SERVER prefix rewrite for --dataset')
    p.add_argument(
        '--format', default='auto', choices=['auto', 'yolo', 'coco', 'openprocessor_export']
    )
    p.add_argument('--accept-suggestions', action='store_true')
    p.add_argument('--map', action='append', default=[], metavar='CLASS=ID')
    p.add_argument('--create', action='append', default=[], metavar='CLASS[=NAME]')
    p.add_argument('--skip', action='append', default=[], metavar='CLASS')
    p.add_argument('--region', action='append', default=[], metavar='CLASS')
    p.add_argument('--images-only', action='store_true')
    p.add_argument('--processing', default='none', choices=['none', 'propose'])
    p.add_argument('--label-trust', default='validated', choices=['validated', 'suggestion'])
    p.add_argument('--parents', default='auto', choices=['auto', 'labels', 'detect'])
    p.add_argument('--missing-label', default='unlabeled', choices=['unlabeled', 'negative'])
    p.add_argument('--freeze-test-split', dest='freeze', action='store_true', default=None)
    p.add_argument('--no-freeze-test-split', dest='freeze', action='store_false')
    p.add_argument('--name', default='cli_import')
    p.add_argument('--force', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='Preview only')
    p.add_argument('--resume', default=None, metavar='IMPORT_ID', help='Resume an import')
    p.add_argument('--state-dir', type=Path, default=None)
    p.add_argument('--poll-interval', type=float, default=2.0)
    p.add_argument('--timeout', type=float, default=7 * 24 * 3600.0)
    return p


def apply_path_map(path: str, path_map: str | None) -> str:
    if not path_map:
        return path
    local, sep, server = path_map.partition('=')
    if not sep or not local or not server:
        raise ImportCliError('--path-map must be LOCAL=SERVER')
    return server + path[len(local) :] if path.startswith(local) else path


def build_mapping(args: argparse.Namespace, dataset_classes: list[str]) -> list[dict[str, Any]]:
    """The explicit mapping entries the flags describe. ``--images-only``
    skips every class the preview found."""
    entries: dict[str, dict[str, Any]] = {}

    def put(name: str, entry: dict[str, Any]) -> None:
        if name in entries:
            raise ImportCliError(f'class {name!r} is given more than once')
        entries[name] = {'dataset_class': name, **entry}

    if args.images_only:
        for name in dataset_classes:
            put(name, {'action': 'skip'})
    for spec in args.map:
        name, sep, class_id = spec.partition('=')
        if not sep or not class_id.isdigit():
            raise ImportCliError(f'--map expects CLASS=ID, got {spec!r}')
        put(name, {'action': 'map', 'class_id': int(class_id)})
    for spec in args.create:
        name, _, new_name = spec.partition('=')
        put(name, {'action': 'create', 'new_class_name': new_name or name})
    for name in args.skip:
        put(name, {'action': 'skip'})
    for name in args.region:
        put(name, {'action': 'region'})
    return list(entries.values())


def _request_body(args: argparse.Namespace, mapping: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        'source': {'path': apply_path_map(args.dataset, args.path_map), 'format': args.format},
        'mapping': mapping,
        'accept_suggestions': args.accept_suggestions,
        'options': {
            'processing': args.processing,
            'label_trust': args.label_trust,
            'parents': args.parents,
            'missing_label': args.missing_label,
            'freeze_test_split': args.freeze,
            'name': args.name,
            'force': args.force,
        },
    }


def _check(resp: httpx.Response) -> dict[str, Any]:
    if resp.status_code >= 400:
        detail = (
            resp.json().get('detail')
            if resp.headers.get('content-type', '').startswith('application/json')
            else resp.text
        )
        raise ImportCliError(
            f'{resp.request.method} {resp.request.url.path}: {resp.status_code} {detail}'
        )
    return resp.json()


def write_ingested(client: httpx.Client, base: str, import_id: str, state_dir: Path) -> int:
    """``ingested/<split>.jsonl`` from the import's ledger entries."""
    out = state_dir / 'ingested'
    out.mkdir(parents=True, exist_ok=True)
    rows: dict[str, list[dict[str, Any]]] = {}
    page = 1
    while True:
        body = _check(
            client.get(
                f'{base}/datasets/imports/{import_id}/entries',
                params={'page': page, 'page_size': 500},
            )
        )
        for e in body['items']:
            if e['status'] != 'ok':
                continue
            rows.setdefault(e.get('split') or 'unsplit', []).append(
                {
                    'image': e['rel_path'],
                    'server_path': e.get('image_path'),
                    'image_id': e.get('image_id'),
                    'status': 'success' if e.get('image_created') else 'duplicate',
                    'positive': e.get('label_state') == 'labeled',
                }
            )
        if page * 500 >= body['total']:
            break
        page += 1
    for split, items in rows.items():
        (out / f'{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in items))
    return sum(len(v) for v in rows.values())


def poll(
    client: httpx.Client, base: str, import_id: str, *, interval: float, timeout: float
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last = ''
    while True:
        job = _check(client.get(f'{base}/datasets/imports/{import_id}'))
        prog = job['progress']
        line = f'{job["status"]}: {prog["images_done"]}/{prog["images_total"]} images'
        if line != last:
            logger.info(line)
            last = line
        if job['status'] in _TERMINAL:
            return job
        if time.monotonic() > deadline:
            raise ImportCliError(f'timed out waiting for {import_id}')
        time.sleep(interval)


def run(args: argparse.Namespace, client: httpx.Client) -> int:
    base = f'{args.api_base.rstrip("/")}/projects/{args.project}'
    if args.resume:
        job = _check(client.post(f'{base}/datasets/imports/{args.resume}/resume'))
        return _finish(args, client, base, job)
    preview = _check(client.post(f'{base}/datasets/preview', json=_request_body(args, [])))
    classes = [c['dataset_class'] for c in preview['classes']]
    logger.info(
        'preview: %s, %d images, %d boxes, %d class(es)',
        preview['format'],
        preview['totals']['images'],
        preview['totals']['boxes'],
        len(classes),
    )
    for c in preview['classes']:
        s = c['suggestion']
        logger.info(
            '  %s: %d boxes; suggestion %s (%s)',
            c['dataset_class'],
            c['boxes'],
            s['action'],
            s['match'],
        )
    for issue in preview['issues']:
        logger.info('  [%s] %s x%d', issue['severity'], issue['code'], issue['count'])
    if args.dry_run:
        return 1 if preview['blocking'] else 0
    body = _request_body(args, build_mapping(args, classes))
    body['expected_import_key'] = None
    resp = client.post(f'{base}/datasets/imports', json=body)
    if resp.status_code == 409 and resp.json().get('detail', {}).get('error') == 'import_resumable':
        import_id = resp.json()['detail']['import_id']
        logger.info('resuming interrupted import %s', import_id)
        job = _check(client.post(f'{base}/datasets/imports/{import_id}/resume'))
    else:
        job = _check(resp)
        if job.get('reused'):
            logger.info('already imported as %s (nothing written)', job['import_id'])
    return _finish(args, client, base, job)


def _finish(args: argparse.Namespace, client: httpx.Client, base: str, job: dict[str, Any]) -> int:
    job = poll(client, base, job['import_id'], interval=args.poll_interval, timeout=args.timeout)
    logger.info('%s: %s', job['import_id'], json.dumps(job['report']))
    if args.state_dir is not None and job['status'] in {'completed', 'completed_with_errors'}:
        n = write_ingested(client, base, job['import_id'], args.state_dir)
        logger.info('wrote %d ingested rows under %s/ingested', n, args.state_dir)
    return 0 if job['status'] in {'completed', 'completed_with_errors'} else 1


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    args = build_parser().parse_args(argv)
    bind_script_project(args.project)
    try:
        with httpx.Client(timeout=60.0) as client:
            return run(args, client)
    except (ImportCliError, httpx.HTTPError) as exc:
        logger.error('%s', exc)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
