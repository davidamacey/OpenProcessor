#!/usr/bin/env python3
"""Bulk ingest of a folder, a manifest, an S3 bucket/prefix or a registered source,
driven through the API.

Starts a run on ``POST /curation/projects/{project}/ingest/bulk``, shows live
progress and exits with the run's outcome. The run itself (bounded concurrent
batches through the ingest pipeline, a ledger that makes it resumable, cancel)
lives in the API; this script is a client, exactly like Cropwright, so the two
cannot diverge.

Like every curation script it binds ``--project`` through the project registry
(OpenSearch), so run it inside the ``api`` container or set ``OPENSEARCH_URL``.

The API reads the files, so ``--folder`` / ``--manifest`` are paths inside the API
container, under a configured source root (``OP_SOURCE_ROOT``). A manifest is a
text file with one image path per line (or JSON Lines with a ``path`` field).
``--s3 s3://bucket/prefix`` ingests objects the API fetches itself; the bucket and
prefix must lie inside the deployment's ``OP_S3_ALLOWED`` and the credentials come from
the API's environment, never from this command. ``--source-id`` runs a source registered
in Cropwright (or ``POST .../ingest/sources``) with its own options.

Re-running a source is idempotent: images already ingested are recognised by
content hash. Starting a source whose earlier run was interrupted, cancelled or
finished with failures resumes that run and skips what it already stored.
Ctrl-C asks the server to cancel (images already stored stay stored) and waits
for it to stop.

Usage (inside the ``api`` container the API listens on port 8000, so pass
``--api-base``; on the host the default ``localhost:4603`` works):
    docker compose exec api python scripts/curation/bulk_ingest.py \\
        --folder /data/source/my_dataset --project cars \\
        --api-base http://localhost:8000/curation

    docker compose exec api python scripts/curation/bulk_ingest.py \\
        --manifest /data/source/list.txt --batch-size 16 --concurrency 8 \\
        --source census --project cars --api-base http://localhost:8000/curation

    docker compose exec api python scripts/curation/bulk_ingest.py \\
        --s3 s3://my-bucket/incoming/ --project cars --api-base http://localhost:8000/curation

    docker compose exec api python scripts/curation/bulk_ingest.py --project cars \\
        --status bulk_20261011T120000_ab12cd34 --api-base http://localhost:8000/curation
    docker compose exec api python scripts/curation/bulk_ingest.py --project cars \\
        --cancel bulk_20261011T120000_ab12cd34 --api-base http://localhost:8000/curation

``--preview`` with any source walks it read-only and prints the file count, bytes,
extension breakdown, image-size sample, estimated time and index footprint, without
starting a run (the walk is capped by the server and reports ``PARTIAL`` when it was).

Exit status: 0 completed, 1 completed with failed images, 2 failed (the run
stopped), 130 cancelled.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import httpx


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from src.config import get_curation_config
from src.services.projects.script_binding import add_project_argument, bind_script_project


TERMINAL = frozenset({'completed', 'completed_with_errors', 'failed', 'cancelled', 'interrupted'})
EXIT_CODES = {'completed': 0, 'completed_with_errors': 1, 'failed': 2, 'cancelled': 130}
REQUEST_TIMEOUT_S = 60.0


class ApiError(Exception):
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument('--folder', help='Directory to walk (a path inside the API container)')
    what.add_argument('--manifest', help='Text file of image paths (inside the API container)')
    what.add_argument('--s3', metavar='S3_URI', help='s3://bucket[/prefix] to list and fetch')
    what.add_argument('--source-id', help='Run a registered ingest source')
    what.add_argument('--status', metavar='RUN_ID', help='Show a run once and exit')
    what.add_argument('--cancel', metavar='RUN_ID', help='Ask a live run to stop and exit')
    parser.add_argument(
        '--api-base',
        default=f'http://localhost:4603{get_curation_config().api_prefix}',
        help='Curation API mount (no trailing slash); requests go to '
        '<api-base>/projects/<--project>/...',
    )
    parser.add_argument(
        '--preview',
        action='store_true',
        help='Only report what the source holds and what ingesting it would cost (file count, '
        'bytes, extensions, image sizes, estimated time and index size), then exit',
    )
    parser.add_argument('--source', default='bulk_ingest', help='source tag for ingested images')
    parser.add_argument(
        '--extensions', help='Comma-separated file extensions for --folder (default: jpg,jpeg,png)'
    )
    parser.add_argument('--batch-size', type=int, help='Images per pipeline batch (server default)')
    parser.add_argument(
        '--concurrency', type=int, help='Batches in flight at once (server default)'
    )
    parser.add_argument(
        '--poll-interval', type=float, default=2.0, help='Seconds between progress reads'
    )
    parser.add_argument('--no-wait', action='store_true', help='Start the run and exit')
    parser.add_argument(
        '--show-failures',
        type=int,
        default=20,
        metavar='N',
        help='Print up to N failed images when the run ends (default 20)',
    )
    add_project_argument(parser)
    return parser


def _call(client: httpx.Client, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
    try:
        resp = client.request(method, url, timeout=REQUEST_TIMEOUT_S, **kwargs)
    except httpx.HTTPError as exc:
        raise ApiError(f'{method} {url}: {exc}') from exc
    if resp.status_code >= 400:
        try:
            detail = resp.json().get('detail')
        except ValueError:
            detail = resp.text
        message = detail.get('message') if isinstance(detail, dict) else detail
        raise ApiError(f'{resp.status_code} {message}')
    result: dict[str, Any] = resp.json()
    return result


def _start_body(args: argparse.Namespace) -> dict[str, Any]:
    body: dict[str, Any] = {} if args.preview else {'source': args.source}
    if args.folder:
        body['folder'] = args.folder
    elif args.manifest:
        body['manifest'] = args.manifest
    elif args.s3:
        if not args.s3.startswith('s3://'):
            raise ApiError(f'--s3 takes s3://bucket[/prefix], got {args.s3!r}')
        bucket, _, prefix = args.s3.removeprefix('s3://').partition('/')
        body['s3'] = {'bucket': bucket, 'prefix': prefix}
    else:
        body['source_id'] = args.source_id
    if args.extensions:
        body['extensions'] = [e for e in args.extensions.split(',') if e.strip()]
    if args.preview:
        return body
    if args.batch_size:
        body['batch_size'] = args.batch_size
    if args.concurrency:
        body['concurrency'] = args.concurrency
    return body


def _fmt_eta(seconds: float | None) -> str:
    if seconds is None:
        return '--'
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours}h{minutes:02d}m' if hours else f'{minutes}m{secs:02d}s'


def _fmt_bytes(count: float) -> str:
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if abs(count) < 1024 or unit == 'TiB':
            return f'{count:.0f} {unit}' if unit == 'B' else f'{count:.1f} {unit}'
        count /= 1024
    return f'{count:.1f} TiB'


def _fmt_duration(low: float, high: float) -> str:
    if round(low) == round(high):
        return f'about {low:.1f} s' if low < 10 else f'about {_fmt_eta(low)}'
    return f'{_fmt_eta(low)} to {_fmt_eta(high)}'


def print_preview(preview: dict[str, Any], config: dict[str, Any]) -> None:
    """Render a ``POST .../ingest/bulk/preview`` answer; every figure is the server's."""
    print(f'{preview["kind"]} {preview["target"]}')
    print(f'  files: {preview["files"]}, {_fmt_bytes(preview["total_bytes"])}')
    for ext, totals in sorted(preview['by_extension'].items()):
        print(f'    {ext}: {totals["files"]} files, {_fmt_bytes(totals["bytes"])}')
    mp = preview['megapixels']
    if mp['sampled']:
        shares = ', '.join(f'{k} MP: {v}' for k, v in mp['histogram'].items())
        gate = mp['share_at_or_above_gate'] * 100
        print(
            f'  image sizes (header sample of {mp["sampled"]}): {shares}; '
            f'{gate:.0f}% at or above {mp["gate_megapixels"]:g} MP'
        )
    skipped = [
        (preview['unreadable_dirs'], 'unreadable directories'),
        (preview['symlinks_skipped'], 'symlinks (not followed)'),
        (preview['missing'], 'missing manifest entries'),
        (preview['outside_roots'], 'entries outside the source roots'),
        (preview['unreadable_files'], 'files that could not be read'),
    ]
    for count, label in skipped:
        if count:
            print(f'  {label}: {count}')
    for path in preview['unreadable_samples']:
        print(f'    unreadable: {path}')
    est = preview['estimate']
    print(
        f'  estimated ingest time: {_fmt_duration(est["duration_s_min"], est["duration_s_max"])}'
        f' ({est["basis"]})'
    )
    print(
        f'  estimated index size: {_fmt_bytes(est["opensearch_bytes"])}, '
        f'about {est["vectors"]:.0f} vectors'
    )
    print(f'  basis: measured figures in {config["preview"]["source_document"]}, not a promise')
    if preview['partial']:
        print(
            f'  PARTIAL ({preview["stop_reason"]}): stopped early, so every figure above is a '
            'lower bound'
        )


def progress_line(run: dict[str, Any]) -> str:
    p = run['progress']
    seen = f'{p["discovered"]}' + ('' if p['discovery_complete'] else '+')
    rate = p['images_per_s_recent'] or p['images_per_s'] or 0.0
    return (
        f'{run["status"]:<22} {p["processed"]}/{seen} '
        f'ingested {p["ingested"]} duplicate {p["duplicates"]} failed {p["failed"]} '
        f'| {rate:.1f} img/s eta {_fmt_eta(p["eta_s"])}'
    )


def print_summary(run: dict[str, Any], failures: list[dict[str, Any]], failed_total: int) -> None:
    p = run['progress']
    print(f'run {run["run_id"]}: {run["status"]}')
    print(
        f'  discovered {p["discovered"]}, ingested {p["ingested"]}, duplicates {p["duplicates"]}, '
        f'failed {p["failed"]}, skipped (already stored) {p["skipped"]}'
    )
    print(
        f'  crops {p["crops_indexed"]}, embedded {p["n_embedded"]}, '
        f'not embedded {p["n_not_embedded"]} ({p["n_embed_failed"]} encoder failures)'
    )
    if p['n_not_embedded']:
        print(
            f'  warning: {p["n_not_embedded"]} stored items have no embedding (encoder failed, not '
            'selected or deferred); they are excluded from clustering, search and the VLM'
        )
    print(f'  {p["images_per_s"] or 0:.2f} img/s over {p["elapsed_s"] or 0:.0f}s')
    stages = ', '.join(
        f'{k} {v:.1f}s' for k, v in sorted(run['stage_s'].items(), key=lambda kv: -kv[1])
    )
    if stages:
        print(f'  stage seconds: {stages}')
    if run.get('error'):
        print(f'  error: {run["error"]}')
    for row in failures:
        print(f'  FAILED {row["path"]} [{row["error_kind"]}] {row["error"]}')
    if failed_total > len(failures):
        print(
            f'  ... and {failed_total - len(failures)} more (GET .../ingest/bulk/{run["run_id"]}/failures)'
        )
    if run['status'] in ('failed', 'cancelled', 'interrupted', 'completed_with_errors'):
        print('  start the same source again to resume: images already stored are skipped')


def _watch(client: httpx.Client, base: str, run: dict[str, Any], interval: float) -> dict[str, Any]:
    url = f'{base}/ingest/bulk/{run["run_id"]}'
    cancelled = False
    while run['status'] not in TERMINAL:
        print('\r' + progress_line(run).ljust(100), end='', flush=True)
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            if cancelled:
                raise
            cancelled = True
            print('\ncancelling after the batches in flight (Ctrl-C again to detach)...')
            _call(client, 'POST', f'{url}/cancel')
        run = _call(client, 'GET', url)
    print('\r' + progress_line(run).ljust(100))
    return run


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.preview and (args.status or args.cancel):
        parser.error('--preview needs a source (--folder, --manifest, --s3 or --source-id)')
    bind_script_project(args.project)
    base = f'{args.api_base.rstrip("/")}/projects/{args.project}'
    with httpx.Client() as client:
        try:
            if args.cancel:
                run = _call(client, 'POST', f'{base}/ingest/bulk/{args.cancel}/cancel')
                print(progress_line(run))
                return 0
            if args.status:
                run = _call(client, 'GET', f'{base}/ingest/bulk/{args.status}')
                print(progress_line(run))
                return EXIT_CODES.get(run['status'], 0)
            if args.preview:
                preview = _call(
                    client, 'POST', f'{base}/ingest/bulk/preview', json=_start_body(args)
                )
                print_preview(preview, _call(client, 'GET', f'{base}/ingest/bulk/config'))
                return 0
            run = _call(client, 'POST', f'{base}/ingest/bulk', json=_start_body(args))
            verb = 'resumed' if run['resumed'] else 'started'
            print(f'{verb} run {run["run_id"]} on {run["source"]["kind"]} {run["source"]["path"]}')
            if args.no_wait:
                return 0
            run = _watch(client, base, run, args.poll_interval)
            page = _call(
                client,
                'GET',
                f'{base}/ingest/bulk/{run["run_id"]}/failures',
                params={'limit': max(args.show_failures, 1)},
            )
        except ApiError as exc:
            print(f'error: {exc}', file=sys.stderr)
            return 2
    print_summary(run, page['items'][: args.show_failures], page['total'])
    return EXIT_CODES.get(run['status'], 2)


if __name__ == '__main__':
    sys.exit(main())
