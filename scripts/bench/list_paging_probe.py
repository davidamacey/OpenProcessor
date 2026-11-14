"""Depth probe for the Detections list: ``GET /crops`` and ``GET /table?rows=detections``.

Walks the list by ``next_cursor`` to ``--depth`` rows and records, per page, the wall time, the rows
returned and the server's own timing header when it sends one (``X-Process-Time`` or
``Server-Timing``). Prints p50 and p95 per ``--band`` rows (default 1,000), so a cost that grows with
depth shows as a rising column. Only reads; nothing is created or deleted.

    list_paging_probe.py --api http://localhost:4603 --project SLUG --depth 10000 --limit 100 \\
        --route crops --route table --out run.json

``--param K=V`` adds a query parameter (a filter or ``sort``) to every request. The 1-minute load
average is recorded before and after each walk; a walk above load 16 is marked noisy.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import httpx


PREFIX = os.environ.get('OP_API_PREFIX', '/curation')
NOISY_LOAD = 16.0
ROUTES = {'crops': ('crops', 'crops', {}), 'table': ('table', 'items', {'rows': 'detections'})}


def _load1() -> float:
    return float(Path('/proc/loadavg').read_text().split()[0])


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def _server_ms(resp: httpx.Response) -> float | None:
    raw = resp.headers.get('x-process-time')
    if raw:
        try:
            return float(raw.rstrip('ms'))
        except ValueError:
            return None
    for part in resp.headers.get('server-timing', '').split(','):
        if 'dur=' in part:
            return float(part.split('dur=')[1].split(';')[0])
    return None


def walk(
    client: httpx.Client, url: str, rows_key: str, params: dict[str, str], depth: int, limit: int
) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    cursor: str | None = None
    served = 0
    while served < depth:
        query = {**params, 'limit': str(limit), **({'cursor': cursor} if cursor else {})}
        t0 = time.perf_counter()
        resp = client.get(url, params=query)
        wall_ms = (time.perf_counter() - t0) * 1000
        resp.raise_for_status()
        body = resp.json()
        rows = len(body[rows_key])
        pages.append(
            {
                'start_row': served,
                'rows': rows,
                'wall_ms': wall_ms,
                'server_ms': _server_ms(resp),
                'bytes': len(resp.content),
                'total': body.get('total'),
            }
        )
        served += rows
        cursor = body.get('next_cursor')
        if not cursor or not rows:
            break
    return pages


def bands(pages: list[dict[str, Any]], band: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for start in range(0, max((p['start_row'] for p in pages), default=0) + 1, band):
        wall = [p['wall_ms'] for p in pages if start <= p['start_row'] < start + band]
        if not wall:
            continue
        out.append(
            {
                'rows_from': start,
                'pages': len(wall),
                'p50_ms': statistics.median(wall),
                'p95_ms': _pct(wall, 0.95),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api', required=True)
    parser.add_argument('--project', required=True)
    parser.add_argument('--depth', type=int, default=10_000)
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--band', type=int, default=1000)
    parser.add_argument('--route', action='append', choices=sorted(ROUTES), default=[])
    parser.add_argument('--param', action='append', default=[], metavar='K=V')
    parser.add_argument('--out')
    args = parser.parse_args()
    extra = dict(p.split('=', 1) for p in args.param)
    report: dict[str, Any] = {'api': args.api, 'project': args.project, 'depth': args.depth}
    with httpx.Client(timeout=300) as client:
        for name in args.route or ['crops']:
            path, rows_key, fixed = ROUTES[name]
            url = f'{args.api}{PREFIX}/projects/{args.project}/{path}'
            load_start = _load1()
            pages = walk(client, url, rows_key, {**fixed, **extra}, args.depth, args.limit)
            load_end = _load1()
            result = {
                'load1_start': load_start,
                'load1_end': load_end,
                'noisy': max(load_start, load_end) > NOISY_LOAD,
                'rows': sum(p['rows'] for p in pages),
                'seconds': sum(p['wall_ms'] for p in pages) / 1000,
                'bands': bands(pages, args.band),
                'pages': pages,
            }
            report[name] = result
            print(
                f'{name}: {result["rows"]} rows in {result["seconds"]:.1f}s (noisy={result["noisy"]})'
            )
            for b in result['bands']:
                print(
                    f'  from row {b["rows_from"]:>7}: p50 {b["p50_ms"]:8.1f} ms  p95 {b["p95_ms"]:8.1f} ms'
                )
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
