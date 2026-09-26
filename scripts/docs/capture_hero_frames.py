#!/usr/bin/env python3
"""Capture the frames for the docs-site hero walkthrough GIF.

Writes same-size 1600x1000 PNG frames, in playback order, into ``--out``;
``scripts/docs/create-workflow-gif.sh`` stitches them into the GIF.

Frames:

1. A terminal session against the API: ``/detect`` on a public COCO
   image, ``/embed/text``, then the curation ingest limits and status
   polling. The commands run for real (stdlib HTTP, no curl subprocess),
   and each step is rendered as a terminal-styled HTML page in the same
   headless browser, so no terminal recorder has to be installed.
2. FastAPI's Swagger UI at ``/docs``: the endpoint groups, then one
   read-only endpoint expanded and executed with its response.

Every request is read-only (inference or GET): nothing is ingested,
labeled or deleted. Capture only from a stack holding public sample data;
the script refuses any API port in 4600-4799.

Needs Playwright with Chromium (``pip install playwright && playwright
install chromium`` inside a venv).
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path


# Public COCO val2017 image from scripts/datasets/manifests/coco_va_200.json
# (Flickr "Attribution License"), fetched from the COCO image host.
COCO_FILE = '000000005001.jpg'
COCO_URL = f'http://images.cocodataset.org/val2017/{COCO_FILE}'
WIDTH, HEIGHT = 1600, 1000
# A read-only curation endpoint shown expanded in Swagger UI.
SWAGGER_TAG = 'Curation'
SWAGGER_OP = 'ingest_config_curation_ingest_config_get'
MAX_LINES = 44
API_PREFIX = os.environ.get('OP_API_PREFIX', '/curation').rstrip('/')


def _guard_api(api: str) -> None:
    port = urllib.parse.urlsplit(api).port or 80
    if 4600 <= port <= 4799:
        sys.exit(
            f'refusing to capture from {api}: ports 4600-4799 are reserved for non-public stacks'
        )


def _http(method: str, url: str, body: bytes | None = None, headers: dict | None = None) -> dict:
    # Every call here is read-only, so a transient 5xx (e.g. a cold Triton
    # gRPC channel) is safe to retry a few times.
    for attempt in range(5):
        req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code < 500 or attempt == 4:
                raise
            time.sleep(2)
    raise AssertionError('unreachable')


def _multipart(field: str, filename: str, data: bytes) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    head = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; '
        f'filename="{filename}"\r\nContent-Type: image/jpeg\r\n\r\n'
    ).encode()
    return (
        head + data + f'\r\n--{boundary}--\r\n'.encode(),
        f'multipart/form-data; boundary={boundary}',
    )


def _pick(d: dict, keys: list[str]) -> dict:
    return {k: d[k] for k in keys if k in d}


def _round(obj):
    if isinstance(obj, float):
        return round(obj, 3)
    if isinstance(obj, list):
        return [_round(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    return obj


def terminal_steps(api: str, image: bytes) -> list[tuple[str, dict]]:
    """Run the read-only calls; return (shell command, trimmed JSON) pairs."""
    steps: list[tuple[str, dict]] = []

    body, ctype = _multipart('image', COCO_FILE, image)
    det = _http('POST', f'{api}/detect', body, {'Content-Type': ctype})
    det_view = {
        'num_detections': det.get('num_detections'),
        'detections': _round(
            [
                _pick(x, ['x1', 'y1', 'x2', 'y2', 'confidence', 'class_name'])
                for x in det.get('detections', [])[:2]
            ]
        ),
        'image': det.get('image'),
        'model': _pick(det.get('model') or {}, ['name', 'device']),
        'total_time_ms': det.get('total_time_ms'),
    }
    steps.append((f'curl -s -X POST $API/detect -F image=@{COCO_FILE} | jq', det_view))

    text = {'text': 'a person riding a bicycle'}
    emb = _http(
        'POST', f'{api}/embed/text', json.dumps(text).encode(), {'Content-Type': 'application/json'}
    )
    emb_view = {
        'text': emb.get('text'),
        'embedding': [*_round((emb.get('embedding') or [])[:4]), '...'],
        'dimensions': len(emb.get('embedding') or []),
        'embedding_norm': emb.get('embedding_norm'),
    }
    steps.append(
        (
            "curl -s -X POST $API/embed/text -H 'content-type: application/json' "
            '-d \'{"text": "a person riding a bicycle"}\' | jq',
            emb_view,
        )
    )

    cur = f'{api}{API_PREFIX}'
    shown = f'$API{API_PREFIX}'
    cfg = _http('GET', f'{cur}/ingest/config')
    upload_keys = ['enabled', 'max_images_per_request', 'accepted_extensions', 'persists_bytes']
    steps.append(
        (f'curl -s {shown}/ingest/config | jq .upload', _pick(cfg.get('upload') or {}, upload_keys))
    )

    status = _http('GET', f'{cur}/ingest/status')
    steps.append(
        (f'curl -s {shown}/ingest/status | jq', _pick(status, ['total', 'by_source', 'by_day']))
    )

    # Counts and the drain verdict only; the dependency list names the
    # deployment's own region models.
    hidden = ('region_dependencies', 'observed_at', 'total_time_ms', 'request_id')
    drain_cmd = (
        f"curl -s {shown}/ingest/region_drain | jq 'del(.region_dependencies, .observed_at)'"
    )
    for _ in range(2):
        drain = _http('GET', f'{cur}/ingest/region_drain')
        steps.append((drain_cmd, {k: v for k, v in drain.items() if k not in hidden}))
    return steps


def _terminal_html(lines: list[tuple[str, str]]) -> str:
    body = []
    for kind, text in lines[-MAX_LINES:]:
        esc = html.escape(text)
        if kind == 'cmd':
            body.append(f'<div><span class="p">$</span> <span class="c">{esc}</span></div>')
        elif kind == 'comment':
            body.append(f'<div class="m">{esc}</div>')
        else:
            body.append(f'<div class="o">{esc}</div>')
    return f"""<!doctype html><html><head><style>
body {{ margin:0; background:#09090b; font-family:'DejaVu Sans Mono',Menlo,monospace; }}
.win {{ margin:28px; height:{HEIGHT - 56}px; border:1px solid #27272a; border-radius:10px; background:#111113; overflow:hidden; }}
.bar {{ height:34px; background:#18181b; border-bottom:1px solid #27272a; display:flex; align-items:center; padding-left:14px; gap:8px; }}
.dot {{ width:12px; height:12px; border-radius:50%; background:#3f3f46; }}
.title {{ color:#a1a1aa; font-size:14px; margin-left:12px; }}
.term {{ padding:14px 20px; font-size:17px; line-height:1.38; white-space:pre; color:#d4d4d8; }}
.p {{ color:#60a5fa; }} .c {{ color:#f4f4f5; }} .o {{ color:#a7f3d0; }} .m {{ color:#71717a; }}
</style></head><body><div class="win"><div class="bar"><span class="dot"></span><span class="dot"></span>
<span class="dot"></span><span class="title">OpenProcessor API: detect, embed, curation ingest status</span></div>
<div class="term">{''.join(body)}</div></div></body></html>"""


def capture(api: str, out: Path, cache: Path, mlflow: str | None = None) -> None:
    from playwright.sync_api import sync_playwright

    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob('*.png'):
        old.unlink()
    cache.mkdir(parents=True, exist_ok=True)
    img_path = cache / COCO_FILE
    if not img_path.exists():
        urllib.request.urlretrieve(COCO_URL, img_path)
    steps = terminal_steps(api, img_path.read_bytes())

    with sync_playwright() as p:
        # Software rendering: a host GPU busy with inference can make headless
        # Chromium's GPU compositor refuse screenshots.
        browser = p.chromium.launch(args=['--disable-gpu'])
        page = browser.new_page(viewport={'width': WIDTH, 'height': HEIGHT})
        n = 0
        lines: list[tuple[str, str]] = [
            ('comment', f'# API={api}  (public COCO sample image, read-only calls)')
        ]
        for cmd, result in steps:
            lines.append(('cmd', cmd))
            lines.extend(('out', line) for line in json.dumps(result, indent=2).splitlines())
            term_html = out / '_terminal.html'
            term_html.write_text(_terminal_html(lines))
            page.goto(term_html.resolve().as_uri(), wait_until='load')
            n += 1
            page.screenshot(path=str(out / f'{n:02d}-terminal.png'))
            lines = lines[-12:] if len(lines) > MAX_LINES else lines

        (out / '_terminal.html').unlink(missing_ok=True)

        # The spec is large; Swagger UI renders tag groups progressively.
        page.goto(f'{api}/docs', wait_until='networkidle')
        page.wait_for_selector('.opblock-tag')
        page.wait_for_timeout(4000)
        n += 1
        page.screenshot(path=str(out / f'{n:02d}-swagger-groups.png'))

        # One read-only endpoint via Swagger's deep link (expands it and
        # scrolls to it), then executed with its live response.
        # A fresh page: a hash-only navigation doesn't trigger deep linking.
        page.close()
        page = browser.new_page(viewport={'width': WIDTH, 'height': HEIGHT})
        page.goto(
            f'{api}/docs#/{urllib.parse.quote(SWAGGER_TAG)}/{SWAGGER_OP}', wait_until='networkidle'
        )
        block = page.locator(f'#operations-{SWAGGER_TAG.replace(" ", "_")}-{SWAGGER_OP}')
        try_it = block.get_by_role('button', name='Try it out')
        try_it.wait_for(timeout=90_000)
        try_it.click()
        block.get_by_role('button', name='Execute').click()
        block.locator('.live-responses-table').wait_for(timeout=60_000)
        page.wait_for_timeout(800)
        block.evaluate('el => el.scrollIntoView({block: "start"})')
        n += 1
        page.screenshot(path=str(out / f'{n:02d}-swagger-endpoint.png'))
        block.locator('.live-responses-table').evaluate(
            'el => el.scrollIntoView({block: "center"})'
        )
        page.wait_for_timeout(400)
        n += 1
        page.screenshot(path=str(out / f'{n:02d}-swagger-response.png'))
        if mlflow:
            # Optional: the training profile's MLflow UI, when it has runs.
            page.goto(mlflow, wait_until='networkidle')
            page.wait_for_timeout(2000)
            n += 1
            page.screenshot(path=str(out / f'{n:02d}-mlflow.png'))
        browser.close()
    print(f'captured {n} frames into {out}')


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--api', required=True, help='API base URL of a public-sample-data stack')
    ap.add_argument('--out', type=Path, required=True, help='directory for the PNG frames')
    ap.add_argument(
        '--cache', type=Path, default=Path('cache/docs-hero'), help='where the COCO image is kept'
    )
    ap.add_argument('--mlflow', help='optional MLflow UI URL on the same public-data stack')
    args = ap.parse_args()
    api = args.api.rstrip('/')
    _guard_api(api)
    mlflow = args.mlflow.rstrip('/') if args.mlflow else None
    if mlflow:
        _guard_api(mlflow)
    capture(api, args.out, args.cache, mlflow)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
