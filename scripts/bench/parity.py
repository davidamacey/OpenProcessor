"""Parity capture and compare: the gate every v0.6.0 wave passes before it merges.

    parity.py capture --tag fp32 --images DIR --api-url URL --triton-url URL \\
        --opensearch-url URL --compose-project NAME --engine-label fp32 \\
        [--engine-file PLAN] [--annotations COCO_DIR] [--probe-images 200] [--vlm-crops N \\
        --vlm-classes a,b,c]
    parity.py compare REF_DIR CAND_DIR [--profile NAME] [--gates FILE] [--out DIR] [--strict]

``capture`` ingests the pinned dataset into a fresh, project-scoped ingest, trains clusters once,
and stores every stage output (detector boxes, crop rectangles, PE crop and whole-frame vectors,
cluster fields, labels, optional MobileCLIP / face / OCR probes and VLM labels) under
``artifacts_local/bench/parity/<tag>/`` as ``capture.json`` plus a float16 ``arrays.npz``.
``compare`` measures a candidate capture against a reference and applies the thresholds of
``parity_gates.json``; it exits 0 on PASS, 1 on any FAIL (or any SKIP with ``--strict``), 2 when
the inputs cannot be compared. Large capture files stay out of git; commit only the summary.

Preconditions of ``capture``: a stack whose ingest embedding policy is ``all``, with the
auto-label worker running (compose profile ``curation``; cluster training is its job), and the
pinned images on the local disk. The curation API serves no embedding vectors, so the PE vectors
are read by document id from the project's OpenSearch indexes through ``--opensearch-url``.
Detector output varies slightly between identical runs (about 1 percent of crop rectangles
differ, mostly by a pixel), so capture the reference configuration twice and ``compare`` those two as the noise
floor before reading a candidate's numbers.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx


if TYPE_CHECKING:
    from collections.abc import Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.bench import parity_capture as cap, parity_gates as gates, parity_store as store
from scripts.bench.parity_assemble import build_arrays, load_gt
from scripts.bench.parity_env import engine_record, environment_record
from scripts.curation._project_worker_utils import curation_api_prefix, scoped_url
from scripts.datasets.bench_set import load_pin


DEFAULT_MANIFEST = _REPO_ROOT / 'scripts/datasets/manifests/coco_bench_2000.json'
DEFAULT_OUT = Path('artifacts_local/bench/parity')
DEFAULT_GATES = Path(__file__).with_name('parity_gates.json')


def _slug(tag: str) -> str:
    return f'par-{re.sub(r"[^a-z0-9]+", "-", tag.lower()).strip("-")}-{datetime.now(UTC):%H%M%S}'


def run_capture(
    args: argparse.Namespace, *, transport: httpx.BaseTransport | None = None
) -> store.Capture:
    """Ingest the pinned set into a fresh project and collect every stage output."""
    pin = load_pin(args.manifest)
    rows = pin['images'][: args.limit or None]
    files = [args.images / r['file_name'] for r in rows]
    missing = [f.name for f in files if not f.is_file()]
    if missing:
        raise store.CaptureError(f'{len(missing)} pinned images are missing, first: {missing[0]}')
    dims = {r['file_name']: (r['width'], r['height']) for r in rows}
    slug = args.project or _slug(args.tag)
    base = scoped_url(args.api_url, curation_api_prefix(), slug, '')
    sections = ['items', 'frames', 'clusters']
    timings: dict[str, float] = {}

    def lap(name: str, since: float) -> None:
        timings[name] = time.monotonic() - since
        print(f'{name}: {timings[name]:.0f}s', flush=True)

    with httpx.Client(timeout=httpx.Timeout(3600.0), transport=transport) as client:
        client.post(
            f'{args.api_url}{curation_api_prefix()}/projects',
            json={'slug': slug, 'display_name': slug},
        ).raise_for_status()
        try:
            environment = environment_record(
                client,
                gpu=args.gpu,
                compose_project=args.compose_project,
                api_url=args.api_url,
                triton_url=args.triton_url,
                opensearch_url=args.opensearch_url,
                manifest=args.manifest,
            )
            policy = cap.require_embed_all(client, base)
            started = time.monotonic()
            image_names = cap.ingest_images(
                client, base, files, batch_size=args.batch_size, concurrency=args.concurrency
            )
            lap('ingest', started)
            started = time.monotonic()
            cluster_job = cap.run_clustering(client, base, args.cluster_timeout)
            lap('cluster', started)
            items = cap.fetch_items(client, base)
            items_index, images_index = cap.project_indexes(client, base)
            os_url = args.opensearch_url.rstrip('/')
            vectors = cap.mget_vectors(client, os_url, items_index, [i['id'] for i in items])
            frames = cap.mget_vectors(client, os_url, images_index, list(image_names))
            vlm: dict[str, str] = {}
            if args.vlm_crops:
                sections.append('vlm')
                ids = cap.vlm_sample(items, image_names, args.vlm_crops)
                vlm = cap.label_with_vlm(
                    client, base, ids, args.vlm_classes.split(','), args.batch_size
                )
            probes: dict[str, list[tuple[Any, ...]]] = {}
            if args.probe_images:
                sections += ['clip', 'faces', 'ocr']
                probes = cap.probe_core(
                    client, args.api_url, files[: args.probe_images], concurrency=args.concurrency
                )
        finally:
            if not args.keep_project:
                client.delete(base, params={'confirm': slug}).raise_for_status()

    gt = load_gt(args.annotations, pin) if args.annotations else None
    arrays = build_arrays(
        items,
        image_names=image_names,
        image_dims=dims,
        vectors=vectors,
        frame_vectors=frames,
        gt=gt,
        vlm=vlm,
        probes=probes,
    )
    meta = {
        'tag': args.tag,
        'created_at': datetime.now(UTC).isoformat(),
        'project': slug,
        'project_kept': args.keep_project,
        'manifest': {
            'name': args.manifest.name,
            'sha256': pin['manifest_sha256'],
            'count': len(rows),
            'seed': pin['seed'],
            'annotations_sha256': pin['annotations_sha256'],
        },
        'engine': engine_record(args.engine_label, args.engine_file, args.engine_record),
        'environment': environment,
        'sections': sections,
        'settings': {
            'ingest_embedding_policy': policy,
            'batch_size': args.batch_size,
            'concurrency': args.concurrency,
            'probe_images': args.probe_images,
            'vlm_crops': args.vlm_crops,
            'ground_truth': bool(gt),
        },
        'cluster_job': cluster_job,
        'phase_seconds': timings,
    }
    return store.Capture(meta, arrays)


def cmd_capture(args: argparse.Namespace) -> int:
    out = args.out / args.tag
    if out.exists() and any(out.iterdir()):
        print(f'{out} already holds a capture; choose another --tag or remove it', file=sys.stderr)
        return 2
    capture = run_capture(args)
    store.save_capture(out, capture)
    counts = json.loads((out / store.META_FILE).read_text(encoding='utf-8'))['counts']
    print(f'captured {counts} into {out}')
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    ref, cand = store.load_capture(args.ref), store.load_capture(args.cand)
    spec = gates.load_gates(args.gates)
    metrics = gates.compute_metrics(ref, cand, spec['settings'])
    result = gates.evaluate(metrics, spec, args.profile)
    text = gates.render_markdown(ref.meta, cand.meta, metrics, result)
    print(text)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / 'parity_summary.md').write_text(text, encoding='utf-8')
        summary = {
            'schema': 1,
            'reference': _brief(ref.meta),
            'candidate': _brief(cand.meta),
            'metrics': metrics,
            'evaluation': result,
        }
        (args.out / 'parity_summary.json').write_text(
            json.dumps(summary, indent=2, sort_keys=True, default=float) + '\n', encoding='utf-8'
        )
    return gates.exit_code(result, strict=args.strict)


def _brief(meta: dict[str, Any]) -> dict[str, Any]:
    env = meta['environment']
    return {
        'tag': meta['tag'],
        'created_at': meta['created_at'],
        'engine': meta['engine'],
        'manifest': meta['manifest'],
        'git_sha': env.get('git_sha'),
        'git_dirty': env.get('git_dirty'),
        'gpu': env.get('hardware', {}).get('gpu', {}).get('model'),
        'image_digests': env.get('image_digests'),
        'sections': meta['sections'],
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('capture', help='ingest the pinned set and store every stage output')
    c.add_argument('--tag', required=True, help='capture name; the output directory')
    c.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    c.add_argument('--images', type=Path, required=True, help='directory with the pinned images')
    c.add_argument('--api-url', required=True)
    c.add_argument('--triton-url', required=True)
    c.add_argument('--opensearch-url', required=True)
    c.add_argument('--compose-project', required=True, help='compose project of the stack')
    c.add_argument('--gpu', default='0')
    c.add_argument('--out', type=Path, default=DEFAULT_OUT)
    c.add_argument('--engine-label', required=True, help='precision or build label, e.g. fp16')
    c.add_argument('--engine-file', type=Path, help='the PE plan file: size and sha256 recorded')
    c.add_argument('--engine-record', action='append', default=[], metavar='KEY=VALUE')
    c.add_argument('--annotations', type=Path, help='COCO annotation directory (purity gate)')
    c.add_argument('--limit', type=int, default=0, help='first N pinned images (smoke runs)')
    c.add_argument('--batch-size', type=int, default=32)
    c.add_argument('--concurrency', type=int, default=4, help='parallel upload requests')
    c.add_argument('--cluster-timeout', type=float, default=3600.0, help='seconds')
    c.add_argument('--probe-images', type=int, default=200, help='core-route sample; 0 skips')
    c.add_argument('--vlm-crops', type=int, default=0, help='label about N crops with the VLM')
    c.add_argument('--vlm-classes', default='', help='comma list of class names for the VLM')
    c.add_argument('--project', help='project slug (default: generated)')
    c.add_argument('--keep-project', action='store_true')
    c.set_defaults(func=cmd_capture)
    d = sub.add_parser('compare', help='gate a candidate capture against a reference')
    d.add_argument('ref', type=Path)
    d.add_argument('cand', type=Path)
    d.add_argument('--gates', type=Path, default=DEFAULT_GATES)
    d.add_argument('--profile', default='precision')
    d.add_argument('--out', type=Path, help='write parity_summary.json and .md here')
    d.add_argument('--strict', action='store_true', help='a SKIP counts as a failure')
    d.set_defaults(func=cmd_compare)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == 'capture' and args.vlm_crops and not args.vlm_classes:
        print('--vlm-crops needs --vlm-classes', file=sys.stderr)
        return 2
    try:
        return int(args.func(args))
    except (store.CaptureError, FileNotFoundError, ValueError) as exc:
        print(f'parity: {exc}', file=sys.stderr)
        return 2
    except httpx.HTTPStatusError as exc:
        body = exc.response.text[:500]
        print(
            f'parity: {exc.request.method} {exc.request.url} -> {exc.response.status_code}: {body}',
            file=sys.stderr,
        )
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
