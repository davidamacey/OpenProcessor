#!/usr/bin/env python3
"""Capture the docs-site "backend in action" screenshots.

Two subcommands, both against a stack holding public sample data only:

``traffic``
    Read-only inference load against the API (``/detect``, ``/embed/*``,
    ``/ocr/predict``, ``/faces/*``, ``/analyze``) with public COCO val2017
    images, so the Grafana and Prometheus panels have live data. Nothing is
    ingested, labeled, indexed or deleted.

``capture``
    1600-px-wide PNGs of every tool the stack ships:

    * ``models.png``: ``GET /models/``, rendered as a terminal frame;
    * ``swagger.png``: the Swagger UI endpoint groups at ``/docs``;
    * ``prometheus-targets.png``: Prometheus scrape targets;
    * ``grafana-<uid>.png``: each provisioned Grafana dashboard (dark theme,
      kiosk mode, last 15 minutes);
    * ``mlflow-experiments.png``, ``mlflow-run.png``, ``mlflow-compare.png``;
    * ``opensearch-indices.png``: the ``op_*`` indexes in OpenSearch
      Dashboards' index management.

    Grafana needs a viewer login. The script never reads a ``.env`` file:
    it takes ``OP_GRAFANA_TOKEN`` (a service-account token) or
    ``OP_GRAFANA_USER`` + ``OP_GRAFANA_PASSWORD`` from the environment,
    sends them only to the Grafana origin, and never prints them. Without
    either, the Grafana captures are skipped.

Every URL is refused if its port is in 4600-4799 (ports reserved for
non-public stacks). Needs Playwright with Chromium inside a venv.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import itertools
import json
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_hero_frames import _guard_api, _multipart, _terminal_html


WIDTH = 1600
MANIFEST = Path(__file__).resolve().parents[1] / 'datasets/manifests/coco_va_200.json'
COCO_HOST = 'http://images.cocodataset.org'
# Read-only routes only; /embed/image keeps its default index=false.
TRAFFIC_ROUTES = [
    '/detect',
    '/embed/image',
    '/ocr/predict',
    '/faces/detect',
    '/faces/recognize',
    '/analyze',
]
TEXT_QUERIES = ['a person riding a bicycle', 'a red bus on a city street', 'a dog on a sofa']


def _public_images(cache: Path, n: int) -> list[tuple[str, bytes]]:
    """``n`` Attribution-licensed COCO val2017 images from the manifest."""
    cache.mkdir(parents=True, exist_ok=True)
    rows = [
        r for r in json.loads(MANIFEST.read_text()) if r['license_name'] == 'Attribution License'
    ]
    rows.sort(key=lambda r: r['image_id'])
    out = []
    for row in rows[:: max(1, len(rows) // n)][:n]:
        path = cache / row['file_name']
        if not path.exists():
            urllib.request.urlretrieve(f'{COCO_HOST}/{row["split"]}/{row["file_name"]}', path)
        out.append((row['file_name'], path.read_bytes()))
    return out


def _post(url: str, body: bytes, ctype: str) -> int:
    req = urllib.request.Request(url, data=body, method='POST', headers={'Content-Type': ctype})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            resp.read()
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code


def traffic(api: str, seconds: int, workers: int, cache: Path) -> None:
    images = _public_images(cache, 24)
    deadline = time.monotonic() + seconds
    jobs = itertools.cycle(
        [(route, img) for img in images for route in TRAFFIC_ROUTES] + [('/embed/text', None)]
    )
    counts: dict[str, int] = {}

    def one(job: tuple[str, tuple[str, bytes] | None]) -> tuple[str, int]:
        route, img = job
        if img is None:
            body = json.dumps({'text': random.choice(TEXT_QUERIES)}).encode()
            return route, _post(f'{api}{route}', body, 'application/json')
        body, ctype = _multipart('image', img[0], img[1])
        return route, _post(f'{api}{route}', body, ctype)

    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        pending = {pool.submit(one, next(jobs)) for _ in range(workers)}
        while pending:
            done, pending = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for fut in done:
                route, status = fut.result()
                key = f'{route} {status}'
                counts[key] = counts.get(key, 0) + 1
                if time.monotonic() < deadline:
                    pending.add(pool.submit(one, next(jobs)))
    for key in sorted(counts):
        print(f'{counts[key]:6d}  {key}')


def _get_json(url: str, headers: dict | None = None, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    hdrs = {'Content-Type': 'application/json', **(headers or {})}
    req = urllib.request.Request(url, data=data, headers=hdrs, method='POST' if data else 'GET')
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def _grafana_auth() -> dict | None:
    token = os.environ.get('OP_GRAFANA_TOKEN')
    if token:
        return {'Authorization': f'Bearer {token}'}
    user, pw = os.environ.get('OP_GRAFANA_USER'), os.environ.get('OP_GRAFANA_PASSWORD')
    if user and pw:
        return {'Authorization': 'Basic ' + base64.b64encode(f'{user}:{pw}'.encode()).decode()}
    return None


def _shot(page, path: Path, full_page: bool = False) -> None:
    page.screenshot(path=str(path), full_page=full_page)
    print(f'wrote {path}')


def capture_models(page, api: str, out: Path) -> None:
    models = _get_json(f'{api}/models/').get('models', [])
    # Loaded models first; the rest are in the repository but not loaded
    # (Triton runs in explicit model-control mode).
    models.sort(key=lambda m: (m.get('status') != 'READY', m.get('name', '')))
    keys = ('name', 'status', 'backend', 'max_batch_size')
    lines = [
        ('comment', f'# API={api}  (Triton models behind the API, public sample stack)'),
        (
            'cmd',
            "curl -s $API/models/ | jq -c '.models[] | {name, status, backend, max_batch_size}'",
        ),
    ]
    lines += [('out', json.dumps({k: m.get(k) for k in keys})) for m in models]
    term = out / '_terminal.html'
    term.write_text(_terminal_html(lines, title='OpenProcessor API: Triton model status'))
    page.goto(term.resolve().as_uri(), wait_until='load')
    _shot(page, out / 'models.png')
    term.unlink(missing_ok=True)


def capture_swagger(page, api: str, out: Path) -> None:
    page.goto(f'{api}/docs', wait_until='networkidle')
    page.wait_for_selector('.opblock-tag')
    page.wait_for_timeout(4000)
    # Collapse every group so the page shows the group list, not one route.
    # Swagger UI virtualizes the tag list: collapsing one group renders the
    # next, so collapse until no open group is left.
    for _ in range(200):
        open_tag = page.locator('h3.opblock-tag[data-is-open="true"]').first
        if open_tag.count() == 0:
            break
        open_tag.click()
        page.wait_for_timeout(150)
    page.evaluate('window.scrollTo(0, 0)')
    page.wait_for_timeout(1000)
    _shot(page, out / 'swagger.png')


def capture_prometheus(page, prom: str, out: Path) -> None:
    page.add_init_script("localStorage.setItem('mantine-color-scheme-value', 'dark')")
    page.goto(f'{prom}/targets', wait_until='load')
    page.wait_for_timeout(5000)
    _shot(page, out / 'prometheus-targets.png')


def capture_grafana(browser, grafana: str, out: Path, minutes: int) -> None:
    auth = _grafana_auth()
    if auth is None:
        print('skipping Grafana: set OP_GRAFANA_TOKEN or OP_GRAFANA_USER/OP_GRAFANA_PASSWORD')
        return
    dashboards = _get_json(f'{grafana}/api/search?type=dash-db', headers=auth)
    ctx = browser.new_context(viewport={'width': WIDTH, 'height': 1000}, extra_http_headers=auth)
    page = ctx.new_page()
    for dash in dashboards:
        url = (
            f'{grafana}/d/{dash["uid"]}?orgId=1&theme=dark&kiosk'
            f'&from=now-{minutes}m&to=now&refresh=off'
        )
        page.goto(url, wait_until='load')
        page.wait_for_timeout(8000)
        _shot(page, out / f'grafana-{dash["uid"]}.png', full_page=True)
    ctx.close()


def capture_mlflow(page, mlflow: str, out: Path) -> None:
    exps = _get_json(f'{mlflow}/api/2.0/mlflow/experiments/search', body={'max_results': 100})
    ids = [e['experiment_id'] for e in exps.get('experiments', [])]
    runs = _get_json(
        f'{mlflow}/api/2.0/mlflow/runs/search',
        body={
            'experiment_ids': ids,
            'max_results': 100,
            'order_by': ['attributes.start_time DESC'],
        },
    ).get('runs', [])
    if not runs:
        print('skipping MLflow: no runs')
        return
    # The training runs: the ones that log the most metrics and params.
    runs.sort(
        key=lambda r: (len(r['data'].get('metrics', [])), len(r['data'].get('params', []))),
        reverse=True,
    )
    best = runs[0]
    exp = best['info']['experiment_id']
    pair = [r for r in runs if r['info']['experiment_id'] == exp][:2]
    run_url = f'{mlflow}/#/experiments/{exp}/runs/{best["info"]["run_id"]}'

    page.add_init_script("localStorage.setItem('_mlflow_dark_mode_toggle_enabled', 'true')")
    page.goto(f'{mlflow}/#/experiments/{exp}', wait_until='load')
    page.wait_for_timeout(3000)
    _shot(page, out / 'mlflow-experiments.png')

    page.goto(run_url, wait_until='load')
    page.wait_for_timeout(3000)
    _shot(page, out / 'mlflow-run.png')

    page.goto(f'{run_url}/model-metrics', wait_until='load')
    page.wait_for_timeout(5000)
    _shot(page, out / 'mlflow-run-metrics.png')

    if len(pair) == 2:
        run_ids = json.dumps([r['info']['run_id'] for r in pair])
        q = urllib.parse.urlencode({'runs': run_ids, 'experiments': json.dumps([exp])})
        page.goto(f'{mlflow}/#/compare-runs?{q}', wait_until='load')
        page.wait_for_timeout(4000)
        # The per-run parameter and metric tables, with only the differences.
        # The switches include MLflow's header theme toggle, which flips this
        # page to light: its compare tables only half-apply the dark theme.
        page.get_by_text('Visualizations', exact=True).click()
        page.get_by_text('Run details', exact=True).click()
        for toggle in page.get_by_role('switch').all():
            toggle.click()
        page.evaluate('window.scrollTo(0, 0)')
        page.wait_for_timeout(1000)
        _shot(page, out / 'mlflow-compare.png')


def capture_opensearch(page, osd: str, out: Path) -> None:
    page.goto(
        f'{osd}/app/opensearch_index_management_dashboards#/indices?search=op_&size=20&sortField=index&sortDirection=asc',
        wait_until='load',
    )
    page.wait_for_timeout(8000)
    _shot(page, out / 'opensearch-indices.png')


def capture(args: argparse.Namespace) -> None:
    from playwright.sync_api import sync_playwright

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        # Software rendering: a host GPU busy with inference can make headless
        # Chromium's GPU compositor refuse screenshots.
        browser = p.chromium.launch(args=['--disable-gpu'])

        def fresh():
            return browser.new_context(
                viewport={'width': WIDTH, 'height': 1000}, color_scheme='dark'
            ).new_page()

        capture_models(fresh(), args.api, out)
        capture_swagger(fresh(), args.api, out)
        if args.prometheus:
            capture_prometheus(fresh(), args.prometheus, out)
        if args.grafana:
            capture_grafana(browser, args.grafana, out, args.minutes)
        if args.mlflow:
            capture_mlflow(fresh(), args.mlflow, out)
        if args.osd:
            capture_opensearch(fresh(), args.osd, out)
        browser.close()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('traffic', help='read-only inference load')
    t.add_argument('--api', required=True)
    t.add_argument('--seconds', type=int, default=240)
    t.add_argument('--workers', type=int, default=6)
    t.add_argument('--cache', type=Path, default=Path('cache/docs-hero'))
    c = sub.add_parser('capture', help='screenshot every tool')
    c.add_argument('--api', required=True)
    c.add_argument('--prometheus')
    c.add_argument('--grafana')
    c.add_argument('--mlflow')
    c.add_argument('--osd', help='OpenSearch Dashboards URL')
    c.add_argument('--minutes', type=int, default=15, help='Grafana time range')
    c.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    for name in ('api', 'prometheus', 'grafana', 'mlflow', 'osd'):
        url = getattr(args, name, None)
        if url:
            url = url.rstrip('/')
            _guard_api(url)
            setattr(args, name, url)
    if args.cmd == 'traffic':
        traffic(args.api, args.seconds, args.workers, args.cache)
    else:
        capture(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
