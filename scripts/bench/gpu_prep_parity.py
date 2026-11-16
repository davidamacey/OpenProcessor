#!/usr/bin/env python3
"""Does a change to the frame engine alter what it answers? The one parity tool of the engine
(``frame_engine/`` and the native engine behind the same protocol), plan 17 section 1.2.

``capture`` sends each photo straight to an engine (protocol version 1; ``--plan frame`` is the
detector and the frame vector, ``--plan ingest`` the plan photo ingest sends: detect, the frame
vector, the crop vectors and inline crop JPEGs; so only the engine is measured, not the API) and
keeps every output tensor. ``--models`` names the models the plan calls: ``real``, the null
stand-ins, or the ``capture`` stand-ins that write every input row they receive
(``frame_engine_stand_ins.py``). ``plan`` writes the plan text, so the load generator sends the
very text this tool does. ``compare`` sets two captures side by side, with the floor measured as
the same engine against itself; ``compare-inputs`` decides whether two engines handed the capture
models the same bytes.

    python scripts/bench/gpu_prep_parity.py capture --url <engine host:port> --images list.txt \\
        --plan ingest --models real --concurrency 1 --out before.npz --crop-counts-out counts.json
    python scripts/bench/gpu_prep_parity.py capture ... --out after.npz
    python scripts/bench/gpu_prep_parity.py compare before.npz after.npz \\
        --floor-a python_1.npz --floor-b python_2.npz --in-flight 1
    python scripts/bench/gpu_prep_parity.py compare-inputs capture_python/ capture_native/

At one photo in flight a change that does not touch arithmetic must give identical bytes for
everything the engine controls. Model outputs that depend on the server's batch composition do not
qualify even for an engine against itself, so the vector rows pass on a floor: the same engine
against itself, two runs in the same hold (plan 17 section 1.2).
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np


# ``python scripts/bench/gpu_prep_parity.py`` puts scripts/bench, not the repository, on the path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

OUTPUTS = (
    'status',
    'decode_route',
    'branch_status',
    'size',
    'letterbox',
    'det_boxes',
    'det_scores',
    'det_classes',
    'whole_embedding',
    'crop_embeddings',
    'crop_embedded',
    'blur_sums',
    'crop_jpegs',
)
EXACT = ('status', 'size', 'letterbox', 'det_classes', 'blur_sums')
PLANS = ('frame', 'ingest')
MODEL_SETS = ('real', 'null', 'capture')
#: Crop JPEG sizes may differ by this share where a build or setting cannot match (plan 17 Q6).
JPEG_SIZE_TOLERANCE = 0.005


def read_list(path: Path) -> list[Path]:
    rows = [ln.strip() for ln in path.read_text().splitlines()]
    return [Path(r) for r in rows if r and not r.startswith('#')]


# --------------------------------------------------------------------------- the plan


def build_plan(
    plan: Literal['frame', 'ingest'],
    models: Literal['real', 'null', 'capture'],
    detector: str,
    net_size: int,
    fill: int,
) -> str:
    """The plan text. ``ingest`` is what photo ingest sends, from the API's own builder
    (``src.services.curation.ingest_gpu_decode.ingest_engine_plan``) for the one-grey letterbox
    fill and every class embedded; ``frame`` is that plan without the crop branch and without
    inline crops. ``models`` replaces the model names with the stand-ins' (nothing else moves)."""
    from src.services.curation.ingest_gpu_decode import ingest_engine_plan

    service = SimpleNamespace(
        profile=SimpleNamespace(
            detector_model=detector, input_size=net_size, letterbox_fill=(fill, fill, fill)
        ),
        policy=SimpleNamespace(
            embedding=SimpleNamespace(mode='all', classes=None),
            detect=SimpleNamespace(classes=None, exclude_classes=()),
        ),
    )
    built: Any = ingest_engine_plan(service)  # type: ignore[arg-type]
    if plan == 'frame':
        built = dataclasses.replace(
            built,
            branches=('detect', 'embed_frame'),
            crop_embed=None,
            embed_classes=None,
            crops='none',
        )
    if models != 'real':
        from scripts.bench.frame_engine_stand_ins import model_names

        names = model_names(models)
        built = dataclasses.replace(
            built,
            detector=dataclasses.replace(built.detector, model=names['detector']),
            embed=dataclasses.replace(built.embed, model=names['frame']),
            crop_embed=(
                dataclasses.replace(built.crop_embed, model=names['crop'])
                if built.crop_embed is not None
                else None
            ),
        )
    return str(built.to_json())


# ------------------------------------------------------------------------- capturing


def source_document(path: Path, root: Path, data: bytes) -> str:
    """The ``source`` input of the native engine's read by key (``read_by_key`` in
    ``contracts/json/frame_engine_v1.json``): the key ``source/<path below root>`` and the size and
    ``imohash`` of ``data`` (the Python package the Rust port is held to), written as the Rust
    client writes it (``frame_engine_source_cases.json``, ``built``)."""
    import imohash

    return json.dumps(
        {
            'imohash': imohash.hashfileobject(io.BytesIO(data)).hex(),
            'key': f'source/{path.relative_to(root).as_posix()}',
            'size': len(data),
        },
        sort_keys=True,
    )


async def _one(
    client: Any, data: bytes, plan: str, sem: asyncio.Semaphore, source: str | None = None
) -> dict[str, np.ndarray]:
    """One photo: its bytes as ``image``, or the document ``source`` names it by."""
    from tritonclient.grpc import InferInput

    inputs = []
    first = ('source', source.encode()) if source is not None else ('image', data)
    for name, value in (first, ('plan', plan.encode())):
        inp = InferInput(name, [1], 'BYTES')
        inp.set_data_from_numpy(np.array([value], dtype=np.object_))
        inputs.append(inp)
    async with sem:
        result = await client.infer('frame_engine', inputs, client_timeout=120.0)
    return {n: result.as_numpy(n) for n in OUTPUTS if result.as_numpy(n) is not None}


async def capture(
    url: str, images: list[Path], concurrency: int, plan: str, keys_root: Path | None = None
) -> list[dict[str, np.ndarray]]:
    """Every photo's answer. With ``keys_root`` each photo is named by its key under that
    directory (the engine has it mounted as its source root) instead of being sent."""
    from tritonclient.grpc.aio import InferenceServerClient

    client = InferenceServerClient(url=url)
    sem = asyncio.Semaphore(concurrency)

    def one(path: Path) -> Any:
        data = path.read_bytes()
        source = source_document(path, keys_root, data) if keys_root is not None else None
        return _one(client, data, plan, sem, source)

    try:
        return list(await asyncio.gather(*(one(p) for p in images)))
    finally:
        await client.close()


def save(path: Path, images: list[Path], rows: list[dict[str, np.ndarray]]) -> None:
    arrays: dict[str, np.ndarray] = {'names': np.array([p.name for p in images])}
    for i, row in enumerate(rows):
        for name, arr in row.items():
            arrays[f'{i}/{name}'] = np.array(arr, dtype=object) if arr.dtype == object else arr
    np.savez_compressed(path, allow_pickle=True, **arrays)


def load(path: Path) -> list[dict[str, np.ndarray]]:
    z = np.load(path, allow_pickle=True)
    n = len(z['names'])
    return [{k.split('/', 1)[1]: z[k] for k in z.files if k.startswith(f'{i}/')} for i in range(n)]


def crop_counts(rows: list[dict[str, np.ndarray]]) -> list[int]:
    """Crops each photo had embedded: what ``op-fe-loadgen z`` replays for the ingest plan."""
    return [int(np.count_nonzero(r['crop_embedded'])) if 'crop_embedded' in r else 0 for r in rows]


def write_crop_counts(path: Path, list_file: Path, rows: list[dict[str, np.ndarray]]) -> None:
    """The counts with the SHA-256 of the list file they were recorded for (the figure the load
    generator records for its list), so a count file cannot be replayed against another list."""
    payload = {
        'list_sha256': hashlib.sha256(list_file.read_bytes()).hexdigest(),
        'crops': crop_counts(rows),
    }
    path.write_text(json.dumps(payload))


# ------------------------------------------------------------------------- comparing


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return 1.0 if d == 0 else float(a @ b / d)


def _status(row: dict[str, np.ndarray]) -> str:
    v = row['status'][0]
    return v.decode() if isinstance(v, bytes) else str(v)


def _by_score(row: dict[str, np.ndarray]) -> np.ndarray:
    return (
        np.argsort(-row['det_scores'], kind='stable') if 'det_scores' in row else np.array([], int)
    )


def _same_or_absent(a: dict[str, np.ndarray], b: dict[str, np.ndarray], key: str) -> bool:
    """A tensor both answers carry is equal, or neither carries it."""
    if key not in a or key not in b:
        return key not in a and key not in b
    return bool(np.array_equal(a[key], b[key]))


def _ordered(values: np.ndarray, order: np.ndarray) -> np.ndarray:
    """``values`` in detection-score order; as given when it has no row per detection (an engine
    that answers another count is compared by its count, not by an index that is not there)."""
    return values[order] if len(values) == len(order) else values


def _pixels_equal(x: bytes, y: bytes) -> bool:
    """Two JPEGs decode to the same pixels. Identical bytes need no decode; a file that does not
    decode is a difference, not an error of the comparison."""
    from PIL import Image

    if bytes(x) == bytes(y):
        return True
    try:
        with Image.open(io.BytesIO(bytes(x))) as ia, Image.open(io.BytesIO(bytes(y))) as ib:
            return bool(
                np.array_equal(np.asarray(ia.convert('RGB')), np.asarray(ib.convert('RGB')))
            )
    except (OSError, ValueError):
        return False


def _jpeg_rows(
    a: dict[str, np.ndarray], b: dict[str, np.ndarray], oa: Any, ob: Any
) -> dict[str, Any]:
    ja, jb = _ordered(a['crop_jpegs'], oa), _ordered(b['crop_jpegs'], ob)
    if len(ja) != len(jb):
        return {'crop_jpeg_count_equal': False, 'crop_jpeg_pixels_equal': False}
    pairs = list(zip(ja, jb, strict=True))
    return {
        'crop_jpeg_count_equal': True,
        'crop_jpeg_equal': all(bytes(x) == bytes(y) for x, y in pairs),
        'crop_jpeg_pixels_equal': all(_pixels_equal(x, y) for x, y in pairs),
        'crop_jpeg_size_ratio': max(
            (abs(len(x) - len(y)) / max(len(x), 1) for x, y in pairs), default=0.0
        ),
    }


def _vector_rows(
    a: dict[str, np.ndarray], b: dict[str, np.ndarray], oa: Any, ob: Any
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if 'whole_embedding' in a and 'whole_embedding' in b:
        out['whole_cos'] = _cos(a['whole_embedding'], b['whole_embedding'])
        out['whole_equal'] = bool(np.array_equal(a['whole_embedding'], b['whole_embedding']))
    if 'crop_embeddings' in a and 'crop_embeddings' in b:
        ca, cb = _ordered(a['crop_embeddings'], oa), _ordered(b['crop_embeddings'], ob)
        out['crop_cos_min'] = min((_cos(x, y) for x, y in zip(ca, cb, strict=True)), default=1.0)
        out['crop_vectors'] = len(ca)
        out['crop_vectors_equal'] = sum(
            bool(np.array_equal(x, y)) for x, y in zip(ca, cb, strict=True)
        )
    return out


def compare_pair(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, float | bool]:
    """Differences of one image's two answers."""
    out: dict[str, Any] = {'status_equal': bool(np.array_equal(a['status'], b['status']))}
    if not out['status_equal'] or _status(a) != 'ok':
        return out
    oa, ob = _by_score(a), _by_score(b)
    out['route_equal'] = _same_or_absent(a, b, 'decode_route')
    out['branch_equal'] = _same_or_absent(a, b, 'branch_status')
    out['exact_equal'] = all(
        np.array_equal(
            a[k][oa] if k == 'det_classes' else a[k],
            b[k][ob] if k == 'det_classes' else b[k],
        )
        for k in EXACT
        if k != 'blur_sums'
    )
    out['same_detections'] = a['det_boxes'].shape == b['det_boxes'].shape
    if not out['same_detections']:
        return out
    out['box_max_abs'] = (
        float(np.abs(a['det_boxes'][oa] - b['det_boxes'][ob]).max()) if len(oa) else 0.0
    )
    out['scores_equal'] = bool(np.array_equal(a['det_scores'][oa], b['det_scores'][ob]))
    out['blur_equal'] = bool(
        np.array_equal(a['blur_sums'][np.r_[0, oa + 1]], b['blur_sums'][np.r_[0, ob + 1]])
    )
    out['embedded_equal'] = (
        bool(np.array_equal(a['crop_embedded'][oa], b['crop_embedded'][ob]))
        if 'crop_embedded' in a and 'crop_embedded' in b
        else True
    )
    out.update(_vector_rows(a, b, oa, ob))
    jpegs_here = 'crop_jpegs' in a and 'crop_jpegs' in b
    if jpegs_here:
        out.update(_jpeg_rows(a, b, oa, ob))
    out['crop_jpeg_equal'] = out.get('crop_jpeg_equal', not jpegs_here)
    out['bytes_equal'] = bool(
        out['exact_equal']
        and out['box_max_abs'] == 0.0
        and out['blur_equal']
        and out['crop_jpeg_equal']
        and out['route_equal']
        and out['branch_equal']
        and out['scores_equal']
        and out['embedded_equal']
    )
    return out


def summarize(rows: list[dict[str, float | bool]]) -> dict[str, float]:
    ok = [r for r in rows if 'same_detections' in r]
    same = [r for r in ok if r['same_detections'] and 'box_max_abs' in r]
    vectors = sum(int(r.get('crop_vectors', 0)) for r in same)
    identical = sum(
        1
        for r in same
        if r['exact_equal'] and r['box_max_abs'] == 0.0 and r['scores_equal'] and r['blur_equal']
    )
    return {
        'images': len(rows),
        'status_differs': sum(not r['status_equal'] for r in rows),
        'route_differs': sum(not r['route_equal'] for r in ok),
        'branch_differs': sum(not r['branch_equal'] for r in ok),
        'detections_differ': len(ok) - len(same),
        'detections_identical': identical,
        'bytes_equal': sum(bool(r.get('bytes_equal')) for r in same),
        'blur_differs': sum(not r['blur_equal'] for r in same),
        'embedded_differs': sum(not r['embedded_equal'] for r in same),
        'jpeg_differs': sum(not r['crop_jpeg_equal'] for r in same),
        'jpeg_count_differs': sum(not r.get('crop_jpeg_count_equal', True) for r in same),
        'jpeg_pixels_differ': sum(not r.get('crop_jpeg_pixels_equal', True) for r in same),
        'jpeg_size_ratio_max': max(
            (float(r.get('crop_jpeg_size_ratio', 0.0)) for r in same), default=0.0
        ),
        'box_max_abs': max((float(r['box_max_abs']) for r in same), default=0.0),
        'whole_bit_identical': sum(bool(r.get('whole_equal')) for r in same),
        'whole_cos_min': min(
            (float(r['whole_cos']) for r in same if 'whole_cos' in r), default=1.0
        ),
        'crop_vectors': vectors,
        'crop_vectors_bit_identical': sum(int(r.get('crop_vectors_equal', 0)) for r in same),
        'crop_cos_min': min(
            (float(r['crop_cos_min']) for r in same if 'crop_cos_min' in r), default=1.0
        ),
        'compared': len(same),
    }


def _share(num: float, den: float) -> float:
    return 1.0 if den == 0 else num / den


def verdict(
    summary: dict[str, float], floor: dict[str, float], in_flight: int
) -> list[dict[str, Any]]:
    """The rows of plan 17 section 1.2 for this comparison. Exact rows must have no difference at
    any load; the detections must be exact at one photo in flight and not worse than the floor
    above; the vector rows must be no worse than the floor (the share bit-identical and the
    minimum cosine); crop JPEGs: same count, same decoded pixels, and bytes identical or sizes
    within :data:`JPEG_SIZE_TOLERANCE`."""
    rows: list[dict[str, Any]] = []

    def row(name: str, value: float, bound: float, ok: bool, kind: str) -> None:
        rows.append({'row': name, 'value': value, 'bound': bound, 'kind': kind, 'pass': ok})

    for name in ('status_differs', 'route_differs', 'branch_differs', 'blur_differs',
                 'embedded_differs', 'jpeg_count_differs', 'jpeg_pixels_differ'):  # fmt: skip
        row(name, summary[name], 0, summary[name] == 0, 'exact')
    det = _share(summary['detections_identical'], summary['images'])
    det_floor = _share(floor['detections_identical'], floor['images'])
    if in_flight == 1:
        row('detections_identical_share', det, 1.0, det == 1.0, 'exact at one in flight')
    else:
        row('detections_identical_share', det, det_floor, det >= det_floor, 'floor')
    for name, num, den in (
        ('whole_bit_identical_share', 'whole_bit_identical', 'compared'),
        ('crop_vectors_bit_identical_share', 'crop_vectors_bit_identical', 'crop_vectors'),
    ):
        value, bound = _share(summary[num], summary[den]), _share(floor[num], floor[den])
        row(name, value, bound, value >= bound, 'floor')
    for name in ('whole_cos_min', 'crop_cos_min'):
        row(name, summary[name], floor[name], summary[name] >= floor[name], 'floor')
    bytes_ok = summary['jpeg_differs'] == 0 or summary['jpeg_size_ratio_max'] <= JPEG_SIZE_TOLERANCE
    row('crop_jpeg_bytes_or_size', summary['jpeg_size_ratio_max'], JPEG_SIZE_TOLERANCE, bytes_ok,
        f'bytes identical, else sizes within {JPEG_SIZE_TOLERANCE:.1%}')  # fmt: skip
    return rows


# ---------------------------------------------------------------- the capture models' inputs


def read_index(model_dir: Path) -> list[tuple[int, str, str]]:
    """``(order, sha256, shape)`` of every row a capture model received, in arrival order."""
    rows = []
    for line in (model_dir / 'index.txt').read_text().splitlines():
        order, digest, shape = line.split('\t')
        rows.append((int(order), digest, shape))
    return rows


def _byte_diff(a: bytes, b: bytes) -> dict[str, Any]:
    if len(a) != len(b):
        return {'differing_bytes': max(len(a), len(b)), 'length_a': len(a), 'length_b': len(b)}
    x, y = np.frombuffer(a, np.uint8), np.frombuffer(b, np.uint8)
    at = np.flatnonzero(x != y)
    return {
        'differing_bytes': len(at),
        'max_abs': int(np.abs(x[at].astype(int) - y[at].astype(int)).max()) if len(at) else 0,
        'first_offset': int(at[0]) if len(at) else None,
    }


def compare_model_inputs(dir_a: Path, dir_b: Path, model: str) -> dict[str, Any]:
    """One model's inputs in two captures: equal when the sorted digests of every row are equal;
    otherwise the rows present in one capture only are matched by arrival order and diffed byte
    by byte."""
    a, b = read_index(dir_a / model), read_index(dir_b / model)
    count_a, count_b = Counter(d for _, d, _ in a), Counter(d for _, d, _ in b)
    out: dict[str, Any] = {
        'rows_a': len(a),
        'rows_b': len(b),
        'equal': sorted(count_a.elements()) == sorted(count_b.elements()),
        'only_a': 0,
        'only_b': 0,
        'diffs': [],
    }
    if out['equal']:
        return out
    surplus_a = _surplus(a, count_a, count_b)
    surplus_b = _surplus(b, count_b, count_a)
    out['only_a'], out['only_b'] = len(surplus_a), len(surplus_b)
    for (oa, da), (ob, db) in zip(surplus_a, surplus_b, strict=False):
        diff = _byte_diff(
            (dir_a / model / f'{da}.bin').read_bytes(), (dir_b / model / f'{db}.bin').read_bytes()
        )
        out['diffs'].append({'order_a': oa, 'order_b': ob, **diff})
    return out


def _surplus(
    rows: list[tuple[int, str, str]], mine: Counter[str], theirs: Counter[str]
) -> list[tuple[int, str]]:
    """The rows (order, digest), in arrival order, that the other capture has no match for: of a
    digest ``mine`` holds ``k`` times and ``theirs`` ``j < k`` times, the last ``k - j``."""
    spare = {d: mine[d] - theirs[d] for d in mine if mine[d] > theirs[d]}
    out: list[tuple[int, str]] = []
    for order, digest, _ in reversed(rows):
        if spare.get(digest, 0) > 0:
            spare[digest] -= 1
            out.append((order, digest))
    return sorted(out)


def compare_inputs(dir_a: Path, dir_b: Path) -> dict[str, Any]:
    """Every capture model both directories hold; a model in one only is unequal."""
    models = sorted({p.parent.name for d in (dir_a, dir_b) for p in d.glob('*/index.txt')})
    report: dict[str, Any] = {}
    for model in models:
        if not (dir_a / model / 'index.txt').exists() or not (dir_b / model / 'index.txt').exists():
            report[model] = {'equal': False, 'error': 'captured in one directory only'}
        else:
            report[model] = compare_model_inputs(dir_a, dir_b, model)
    return {'models': report, 'equal': bool(report) and all(m['equal'] for m in report.values())}


# --------------------------------------------------------------------------------- CLI


def _plan_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('--plan', choices=PLANS, default='ingest')
    p.add_argument('--models', choices=MODEL_SETS, default='real')
    p.add_argument('--detector', default='yolov11_small_trt_end2end')
    p.add_argument('--net-size', type=int, default=640)
    p.add_argument('--letterbox-fill', type=int, default=114)


def _plan_of(args: argparse.Namespace) -> str:
    return build_plan(args.plan, args.models, args.detector, args.net_size, args.letterbox_fill)


def _compare_main(args: argparse.Namespace) -> int:
    ra, rb = load(args.a), load(args.b)
    if len(ra) != len(rb):
        print('captures cover different image counts', file=sys.stderr)
        return 2
    summary = summarize([compare_pair(x, y) for x, y in zip(ra, rb, strict=True)])
    for key, value in summary.items():
        print(f'{key:28s} {value}')
    if not (args.floor_a and args.floor_b):
        return 0
    fa, fb = load(args.floor_a), load(args.floor_b)
    floor = summarize([compare_pair(x, y) for x, y in zip(fa, fb, strict=True)])
    rows = verdict(summary, floor, args.in_flight)
    print(json.dumps({'floor': floor, 'verdict': rows}, indent=2))
    return 0 if all(r['pass'] for r in rows) else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    cap = sub.add_parser('capture')
    cap.add_argument('--url', required=True, help='frame engine gRPC host:port')
    _plan_args(cap)
    cap.add_argument('--images', type=Path, required=True, help='file with one image path per line')
    cap.add_argument('--limit', type=int, default=0)
    cap.add_argument('--concurrency', type=int, default=1)
    cap.add_argument('--out', type=Path, required=True)
    cap.add_argument('--crop-counts-out', type=Path, help="record each photo's crop count")
    cap.add_argument(
        '--keys-root',
        type=Path,
        help='native engine only: name each photo by its key under this directory, which the '
        'engine has mounted as its source root, instead of sending its bytes',
    )
    plan = sub.add_parser('plan', help='write the plan text the load generator sends')
    _plan_args(plan)
    plan.add_argument('--out', type=Path, required=True)
    cmp_ = sub.add_parser('compare')
    cmp_.add_argument('a', type=Path)
    cmp_.add_argument('b', type=Path)
    cmp_.add_argument('--floor-a', type=Path, help='the same engine, run one')
    cmp_.add_argument('--floor-b', type=Path, help='the same engine, run two')
    cmp_.add_argument('--in-flight', type=int, default=1, help='the load of the captures')
    inp = sub.add_parser('compare-inputs', help='what two engines handed the capture models')
    inp.add_argument('dir_a', type=Path)
    inp.add_argument('dir_b', type=Path)
    args = ap.parse_args(argv)
    if args.cmd == 'plan':
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(_plan_of(args))
        print(f'plan written to {args.out}')
        return 0
    if args.cmd == 'capture':
        if args.crop_counts_out and args.limit:
            ap.error('--crop-counts-out records the whole list: do not combine it with --limit')
        images = read_list(args.images)[: args.limit or None]
        rows = asyncio.run(
            capture(args.url, images, args.concurrency, _plan_of(args), args.keys_root)
        )
        save(args.out, images, rows)
        if args.crop_counts_out:
            write_crop_counts(args.crop_counts_out, args.images, rows)
        print(f'{len(images)} images captured to {args.out}')
        return 0
    if args.cmd == 'compare-inputs':
        report = compare_inputs(args.dir_a, args.dir_b)
        print(json.dumps(report, indent=2))
        return 0 if report['equal'] else 1
    return _compare_main(args)


if __name__ == '__main__':
    raise SystemExit(main())
