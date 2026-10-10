"""Wave 0 full-pipeline probes on top of the baseline suite.

Phases (``--phases``, comma list), merged into one JSON document like ``baseline_suite.py``:

* ``env``: the baseline environment record plus every GPU's compute processes (with the
  container each one belongs to), the host load and the stack's own containers.
* ``sweep``: ingest over a grid of client settings and Triton arms. ``--sweep-points`` is a list
  of ``<batch>x<concurrency>``; ``--triton-arms`` a ``;``-separated list of arms such as
  ``pe=2;pe=1,det=3`` (instance counts, applied through the model store and reloaded; the shipped
  configs are restored afterwards). One run of the phase per image set gives the high-resolution
  numbers: point ``--manifest`` and ``--images`` at set B.
* ``stages``: per-endpoint cost of the stages ingest does not call (faces, OCR, the combined
  analyze route) and of detection/embedding, as images/s, API and Triton CPU per image, and the
  Triton model time per image.
* ``attrib``: py-spy sampling of the API container while ingest runs, aggregated into where the
  Python-side CPU goes.

    wave0_suite.py --manifest scripts/datasets/manifests/coco_bench_2000.json \\
        --images data/samples/coco_bench_2000/images --out artifacts_local/bench/w0_sweep.json \\
        --project w0tp --api-url http://127.0.0.1:4703 --triton-url http://127.0.0.1:4700 \\
        --opensearch-url http://127.0.0.1:4707 --models-dir /path/to/stack/models \\
        --source-map HOST_DIR=/data/source/coco_bench_2000 --phases env,sweep \\
        --sweep-points 32x1,32x4,32x16 --skip-cluster --skip-settle

Nothing here reads image content into the report; paths and names are not recorded.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx


if TYPE_CHECKING:
    from collections.abc import Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.bench import baseline_suite as bs
from scripts.bench.baseline_report import percentile
from scripts.bench.suite_lib import SCHEMA, summarize_runs, triton_stats_delta
from scripts.bench.suite_probes import cgroup_cpu_s, phase_env, run_cmd, triton_snapshot, wait_idle
from scripts.bench.wave0_lib import aggregate_collapsed, parse_arm, parse_points, triton_gpu_busy
from scripts.bench.wave0_probes import (
    MODEL_KEYS,
    TritonModels,
    Wave0Gauges,
    pyspy_record,
    thread_cpu_census,
)
from scripts.datasets.bench_set import load_pin


PHASES = ('env', 'sweep', 'stages', 'attrib')
STAGE_ENDPOINTS: tuple[tuple[str, str, dict[str, str]], ...] = (
    ('detect', '/detect', {}),
    ('embed_image', '/embed/image', {}),
    ('faces_detect', '/faces/detect', {}),
    ('faces_recognize', '/faces/recognize', {}),
    ('ocr_predict', '/ocr/predict', {}),
    ('analyze', '/analyze', {'enable_ocr': 'false'}),
    ('analyze_ocr', '/analyze', {'enable_ocr': 'true'}),
)
SCHEMA_WAVE0 = f'{SCHEMA}-wave0-1'


def foreign_gpu_processes(project: str) -> list[dict[str, Any]]:
    """Every compute process on every GPU with its container name ('' for a host process)."""
    out = run_cmd(
        [
            'nvidia-smi',
            '--query-compute-apps=gpu_uuid,pid,used_memory,name',
            '--format=csv,noheader,nounits',
        ]
    )
    uuids = {
        ln.split(',')[1].strip(): ln.split(',')[0].strip()
        for ln in run_cmd(
            ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader']
        ).splitlines()
    }
    rows = []
    for line in out.splitlines():
        uuid, pid, mem, name = [p.strip() for p in line.split(',', 3)]
        cgroup = Path(f'/proc/{pid}/cgroup')
        text = cgroup.read_text() if cgroup.is_file() else ''
        container = ''
        for part in text.replace('/', ' ').replace('-', ' ').replace('.', ' ').split():
            if len(part) == 64 and all(c in '0123456789abcdef' for c in part):
                container = run_cmd(['docker', 'inspect', '--format', '{{.Name}}', part]).strip(
                    '/\n'
                )
                break
        rows.append(
            {
                'gpu': uuids.get(uuid, uuid),
                'pid': int(pid),
                'used_mib': float(mem),
                'process': name,
                'container': container,
                'own': container.startswith(f'{project}-'),
            }
        )
    return rows


def phase_wave0_env(args: argparse.Namespace, client: httpx.Client) -> dict[str, Any]:
    first = argparse.Namespace(**{**vars(args), 'gpu': args.gpu.split(',')[0]})
    env = phase_env(first, client)
    env['gpu_processes'] = foreign_gpu_processes(args.project)
    env['gpu_memory_mib'] = {
        ln.split(',')[0].strip(): [float(x) for x in ln.split(',')[1:]]
        for ln in run_cmd(
            [
                'nvidia-smi',
                '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits',
            ]
        ).splitlines()
    }
    env['schema_wave0'] = SCHEMA_WAVE0
    return env


# ---------------------------------------------------------------- sweep


def enrich(run: dict[str, Any]) -> None:
    """Add GPU busy share, host CPU, clocks and foreign load to a run's flat metrics."""
    flat = run['flat']
    wall = flat['wall_s']
    for key, value in triton_gpu_busy(run['triton'], wall).items():
        flat[key] = value
    for gpu, summary in run.get('gpu', {}).items():
        flat[f'gpu{gpu}_util_mean_pct'] = summary.get('util_mean_pct', 0.0)
        flat[f'gpu{gpu}_mem_peak_mb'] = summary.get('mem_peak_mb', 0.0)
    for gpu, att in run.get('attribution_by_gpu', {}).items():
        flat[f'gpu{gpu}_own_sm_mean_pct'] = att.get('own_sm_mean_pct', 0.0)
        flat[f'gpu{gpu}_foreign_sm_mean_pct'] = att.get('foreign_sm_mean_pct', 0.0)
    for gpu, clk in run.get('clocks_by_gpu', {}).items():
        flat[f'gpu{gpu}_sm_mhz_mean'] = clk.get('sm_mhz_mean', 0.0)
        flat[f'gpu{gpu}_throttled_share'] = clk.get('throttled_share', 0.0)
    flat['host_busy_pct'] = run['host_busy_pct']
    flat['load_avg_1m'] = run['load_average'][0]
    for model, row in run['triton'].items():
        flat[f'triton_{model}_mean_batch'] = row['mean_batch']
        flat[f'triton_{model}_infer_ms_per_image'] = (
            row['exec_infer_ms'] * row['executions'] / max(flat['wall_s'] * flat['images_per_s'], 1)
        )
        flat[f'triton_{model}_queue_ms'] = row['queue_ms_per_request']


def point_args(args: argparse.Namespace, batch: int, conc: int) -> argparse.Namespace:
    return argparse.Namespace(
        **{**vars(args), 'batch_size': batch, 'concurrency': conc, 'gauges_cls': Wave0Gauges}
    )


def apply_arm(models: TritonModels, arm: dict[str, int]) -> None:
    """Apply one arm: ``pe=2`` sets an instance count, ``pe_delay=2000`` the queue delay in
    microseconds, ``pe_nopref=1`` removes ``preferred_batch_size`` (model keys from MODEL_KEYS)."""
    per_model: dict[str, dict[str, Any]] = {}
    for key, value in arm.items():
        name, _, knob = key.partition('_')
        opts = per_model.setdefault(MODEL_KEYS[name], {})
        if not knob:
            opts['count'] = value
        elif knob == 'delay':
            opts['queue_delay_us'] = value
        elif knob == 'nopref':
            opts['drop_preferred'] = bool(value)
        else:
            msg = f'unknown arm knob {key!r}'
            raise ValueError(msg)
    for model, opts in per_model.items():
        models.apply(model, **opts)


def phase_sweep(args: argparse.Namespace) -> dict[str, Any]:
    pin = load_pin(args.manifest)
    files = [args.images / r['file_name'] for r in pin['images']][: args.limit or None]
    arms = args.triton_arms.split(';') if args.triton_arms else ['']
    points = parse_points(args.sweep_points)
    out: dict[str, Any] = {
        'config': {
            'images': len(files),
            'modes': args.modes,
            'reps': args.reps,
            'points': [f'{b}x{c}' for b, c in points],
            'arms': arms,
            'dataset_manifest_sha256': pin['manifest_sha256'],
            'cluster_training': not args.skip_cluster,
        },
        'arms': {},
    }
    with httpx.Client(timeout=60.0) as client:
        models = TritonModels(client, args.triton_url, args.models_dir)
        try:
            for arm_spec in arms:
                arm = parse_arm(arm_spec)
                models.restore()  # every arm starts from the shipped configs
                apply_arm(models, arm)
                label = arm_spec or 'shipped'
                modes = args.modes.split(',')
                for mode in modes:
                    bs.ingest_rep(
                        point_args(args, points[0][0], points[0][1]),
                        mode,
                        0,
                        files[: args.warmup],
                        warmup=True,
                    )
                results: dict[str, Any] = {}
                for batch, conc in points:
                    pargs = point_args(args, batch, conc)
                    runs: dict[str, list[dict[str, Any]]] = {m: [] for m in modes}
                    for rep in range(1, args.reps + 1):
                        for mode in modes:
                            run = bs.ingest_rep(pargs, mode, rep, files)
                            enrich(run)
                            runs[mode].append(run)
                    results[f'b{batch}_c{conc}'] = {
                        m: {'summary': summarize_runs([r['flat'] for r in rs]), 'runs': rs}
                        for m, rs in runs.items()
                    }
                    args.out.with_suffix('.partial.json').write_text(json.dumps(out, default=str))
                out['arms'][label] = {'instances': arm, 'points': results}
        finally:
            models.restore()
    return out


# ---------------------------------------------------------------- stages


def phase_stages(args: argparse.Namespace) -> dict[str, Any]:
    if not args.triton_arms:
        return stages_once(args)
    out: dict[str, Any] = {'arms': {}}
    with httpx.Client(timeout=60.0) as client:
        models = TritonModels(client, args.triton_url, args.models_dir)
        try:
            for spec in args.triton_arms.split(';'):
                models.restore()
                apply_arm(models, parse_arm(spec))
                out['arms'][spec or 'shipped'] = stages_once(args)
        finally:
            models.restore()
    return out


def stages_once(args: argparse.Namespace) -> dict[str, Any]:
    pin = load_pin(args.manifest)
    files = [args.images / r['file_name'] for r in pin['images']][: args.endpoint_images]
    blobs = [p.read_bytes() for p in files]
    dims = {r['file_name']: (r['width'], r['height']) for r in pin['images']}
    out: dict[str, Any] = {
        'images': len(blobs),
        'mean_megapixels': statistics.fmean(dims[p.name][0] * dims[p.name][1] / 1e6 for p in files),
        'mean_jpeg_bytes': statistics.fmean(len(b) for b in blobs),
    }
    with httpx.Client(timeout=300.0) as client:
        for name, path, params in STAGE_ENDPOINTS:
            if args.stage_endpoints and name not in args.stage_endpoints.split(','):
                continue
            for threads in (1, args.concurrency):
                runs: list[dict[str, Any]] = []
                wait_idle(args.project)
                for _ in range(args.reps):
                    before = triton_snapshot(client, args.triton_url)
                    cpu0 = {c: cgroup_cpu_s(f'{args.project}-{c}') for c in ('api', 'triton')}
                    lat: list[float] = []
                    failures = 0

                    def one(
                        blob: bytes,
                        path: str = path,
                        lat: list[float] = lat,
                        params: dict[str, str] = params,
                    ) -> None:
                        nonlocal failures
                        t = time.monotonic()
                        r = client.post(
                            f'{args.api_url}{path}',
                            params=params,
                            files={'image': ('i.jpg', blob, 'image/jpeg')},
                        )
                        if r.status_code != 200:
                            failures += 1
                        lat.append(time.monotonic() - t)

                    started = time.monotonic()
                    with (
                        Wave0Gauges(args.project, args.gpu) as gauges,
                        ThreadPoolExecutor(threads) as pool,
                    ):
                        list(pool.map(one, blobs))
                    wall = time.monotonic() - started
                    cpu = {c: cgroup_cpu_s(f'{args.project}-{c}') - cpu0[c] for c in cpu0}
                    delta = triton_stats_delta(before, triton_snapshot(client, args.triton_url))
                    n = len(blobs)
                    flat = {
                        'images_per_s': n / wall,
                        'wall_s': wall,
                        'failed': float(failures),
                        'p50_ms': statistics.median(lat) * 1000,
                        'p95_ms': percentile(lat, 95) * 1000,
                        'api_cpu_s_per_image': cpu['api'] / n,
                        'triton_cpu_s_per_image': cpu['triton'] / n,
                        'host_busy_pct': gauges.result()['host_busy_pct'],
                    }
                    for model, row in delta.items():
                        flat[f'triton_{model}_infer_ms_per_image'] = (
                            row['exec_infer_ms'] * row['executions'] / n
                        )
                        flat[f'triton_{model}_inferences_per_image'] = row['inferences'] / n
                        flat[f'triton_{model}_mean_batch'] = row['mean_batch']
                    runs.append({'flat': flat, 'triton': delta, **gauges.result()})
                out.setdefault(name, {})[f'threads_{threads}'] = {
                    'summary': summarize_runs([r['flat'] for r in runs]),
                    'runs': runs,
                }
    return out


# ---------------------------------------------------------------- attribution


def delayed_census(container: str, delay_s: float, seconds: float) -> dict[str, Any]:
    """Wait for the ingest window to open, then take the thread CPU census."""
    time.sleep(delay_s)
    return thread_cpu_census(container, seconds)


def phase_attrib(args: argparse.Namespace) -> dict[str, Any]:
    pin = load_pin(args.manifest)
    files = [args.images / r['file_name'] for r in pin['images']][: args.limit or None]
    container = f'{args.project}-api'
    image = run_cmd(['docker', 'inspect', '--format', '{{.Config.Image}}', container]).strip()
    out_dir = args.perf_dir / 'pyspy'
    results: dict[str, Any] = {
        'seconds': args.pyspy_seconds,
        'rate_hz': args.pyspy_rate,
        'census_seconds': args.census_seconds,
    }
    pargs = point_args(args, args.batch_size, args.concurrency)
    for mode in args.modes.split(','):
        bs.ingest_rep(pargs, mode, 0, files[: args.warmup], warmup=True)
        # Pass 1, no profiler: CPU seconds by thread name in the middle of the ingest window.
        with ThreadPoolExecutor(1) as pool:
            census = pool.submit(delayed_census, container, args.census_delay, args.census_seconds)
            run = bs.ingest_rep(pargs, mode, 1, files)
            threads = census.result()
        enrich(run)
        cpu_stat = run_cmd(['docker', 'exec', container, 'cat', '/sys/fs/cgroup/cpu.stat'])
        # Pass 2, py-spy on every worker: it costs CPU itself, so its rate is recorded next to
        # the throughput it left (``flat_pyspy``); the stacks, not the rate, are what it adds.
        with ThreadPoolExecutor(1) as pool:
            spy = pool.submit(
                pyspy_record, container, image, args.pyspy_dir, out_dir, mode,
                args.pyspy_seconds, args.pyspy_rate,
            )  # fmt: skip
            time.sleep(5)
            spied = bs.ingest_rep(pargs, mode, 2, files)
            raws = spy.result()
        lines = [ln for raw in raws for ln in raw.read_text().splitlines()]
        results[mode] = {
            'flat': run['flat'],
            'flat_pyspy': spied['flat'],
            'stages': run['stages'],
            'triton': run['triton'],
            'profile': aggregate_collapsed(lines),
            'thread_cpu_by_name': threads,
            'cgroup_cpu_stat_since_start': dict(
                ln.split()
                for ln in cpu_stat.splitlines()
                if ln.split()[0] in ('user_usec', 'system_usec')
            ),
        }
    return results


# ---------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    p = bs.build_parser()
    p.set_defaults(phases='env,sweep', modes='upload', gpu='2', concurrency=4)
    p.add_argument('--models-dir', type=Path, help='the stack model store mounted into Triton')
    p.add_argument('--sweep-points', default='32x4', help='<batch>x<concurrency>, comma list')
    p.add_argument('--triton-arms', default='', help='; list of pe=N,det=N arms (empty: shipped)')
    p.add_argument(
        '--stage-endpoints', default='', help='comma list of endpoint names (default: all)'
    )
    p.add_argument('--pyspy-dir', type=Path, help='directory holding bin/py-spy')
    p.add_argument('--pyspy-seconds', type=int, default=60)
    p.add_argument('--census-seconds', type=int, default=30)
    p.add_argument('--census-delay', type=int, default=20)
    p.add_argument('--pyspy-rate', type=int, default=50)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.perf_dir = args.perf_dir.resolve()
    args.images = args.images.resolve()
    phases = args.phases.split(',')
    if set(phases) - set(PHASES):
        sys.exit(f'--phases must be from {PHASES}')
    doc: dict[str, Any] = (
        json.loads(args.out.read_text()) if args.out.is_file() else {'schema': SCHEMA_WAVE0}
    )
    runners = {
        'sweep': lambda: phase_sweep(args),
        'stages': lambda: phase_stages(args),
        'attrib': lambda: phase_attrib(args),
    }
    with httpx.Client(timeout=30.0) as client:
        for phase in phases:
            started = time.monotonic()
            result = phase_wave0_env(args, client) if phase == 'env' else runners[phase]()
            doc[phase] = result
            doc.setdefault('phase_seconds', {})[phase] = time.monotonic() - started
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(doc, indent=2, sort_keys=True, default=str) + '\n')
            print(f'phase {phase} done in {time.monotonic() - started:.0f}s', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
