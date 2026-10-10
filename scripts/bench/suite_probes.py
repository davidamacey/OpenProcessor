"""Host and stack probes for the baseline suite: processes, GPU gauges, CPU, Triton, environment.

Everything here reads the machine or the running stack (``docker``, ``nvidia-smi``, cgroup
files, Triton and OpenSearch HTTP) and returns plain data; the arithmetic lives in
``suite_lib.py`` and the measurement phases in ``baseline_suite.py``.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence

    import httpx

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.bench.baseline_report import parse_prometheus
from scripts.bench.run_baseline import GpuSampler, read_nvidia_smi
from scripts.bench.suite_lib import foreign_gpu_summary, parse_pmon, parse_triton_stats
from scripts.datasets.bench_set import load_pin


STAGE_PREFIX = 'op_pipeline_stage'


def run_cmd(cmd: list[str], timeout: float = 30.0) -> str:
    """stdout of a command, or an empty string when it is missing or fails."""
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=timeout
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return ''


def own_pids(project: str) -> set[int]:
    """Host pids of every process in the stack's containers."""
    pids: set[int] = set()
    for name in run_cmd(
        ['docker', 'ps', '--filter', f'name=^{project}-', '--format', '{{.Names}}']
    ).split():
        for line in run_cmd(['docker', 'top', name, '-eo', 'pid']).splitlines()[1:]:
            if line.strip().isdigit():
                pids.add(int(line))
    return pids


class PmonSampler:
    """Samples per-process SM utilization of one GPU once a second."""

    def __init__(self, gpu: str, interval: float = 1.0) -> None:
        self.gpu = gpu
        self.interval = interval
        self.samples: list[dict[int, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            out = run_cmd(['nvidia-smi', 'pmon', '-c', '1', '-s', 'u'], timeout=10)
            if out:
                self.samples.append(parse_pmon(out, self.gpu))
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)


class Gauges:
    """GPU memory/util (all GPUs) and per-process SM share, around one measured window."""

    def __init__(self, project: str, gpu: str) -> None:
        self.project, self.gpu = project, gpu
        self.gpus = GpuSampler(read_nvidia_smi, 1.0)
        self.pmon = PmonSampler(gpu)

    def __enter__(self) -> Gauges:
        self.gpus.start()
        self.pmon.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.gpus.stop()
        self.pmon.stop()

    def result(self) -> dict[str, Any]:
        return {
            'gpu': self.gpus.summary(),
            'attribution': foreign_gpu_summary(self.pmon.samples, own_pids(self.project)),
        }


def cgroup_cpu_s(container: str) -> float:
    for line in run_cmd(
        ['docker', 'exec', container, 'cat', '/sys/fs/cgroup/cpu.stat']
    ).splitlines():
        if line.startswith('usage_usec'):
            return int(line.split()[1]) / 1e6
    return 0.0


BACKGROUND = ('api', 'vlm-worker', 'opensearch', 'detection-worker', 'cluster-refresh')


def background_cores(project: str, seconds: float = 10.0) -> dict[str, float]:
    """CPU cores used by each stack container over a short window with no load applied."""
    before = {c: cgroup_cpu_s(f'{project}-{c}') for c in BACKGROUND}
    time.sleep(seconds)
    return {c: (cgroup_cpu_s(f'{project}-{c}') - before[c]) / seconds for c in BACKGROUND}


def wait_idle(project: str, limit_s: float = 300.0, threshold: float = 0.3) -> dict[str, Any]:
    """Block until the API and VLM worker are quiet (below ``threshold`` cores), or ``limit_s``."""
    started = time.monotonic()
    while True:
        cores = background_cores(project)
        quiet = cores['api'] < threshold and cores['vlm-worker'] < threshold
        if quiet or time.monotonic() - started > limit_s:
            return {'cores': cores, 'waited_s': time.monotonic() - started, 'quiet': quiet}


def triton_snapshot(client: httpx.Client, url: str) -> dict[str, Any]:
    return parse_triton_stats(client.get(f'{url}/v2/models/stats').raise_for_status().json())


def triton_configs(client: httpx.Client, url: str, models: list[str]) -> dict[str, Any]:
    return {m: client.get(f'{url}/v2/models/{m}/config').raise_for_status().json() for m in models}


def scrape(client: httpx.Client, url: str) -> dict[str, float]:
    return parse_prometheus(client.get(url).raise_for_status().text)


def stage_table(delta: dict[str, Any]) -> dict[str, dict[str, float]]:
    """``{stage: {calls, seconds, mean_ms, bytes}}`` from an ``op_pipeline_stage_*`` delta."""
    table: dict[str, dict[str, float]] = {}
    for key, h in delta['histograms'].items():
        m = re.fullmatch(rf'{STAGE_PREFIX}_seconds\{{stage=(\w+)\}}', key)
        if m:
            table.setdefault(m[1], {}).update(
                calls=h['count'], seconds=h['sum'], mean_ms=h['mean'] * 1000.0
            )
    for key, v in delta['counters'].items():
        m = re.fullmatch(rf'{STAGE_PREFIX}_bytes_total\{{stage=(\w+)\}}', key)
        if m:
            table.setdefault(m[1], {})['bytes'] = v
    return table


def chunks(items: Sequence[Any], size: int) -> list[Sequence[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


# ---------------------------------------------------------------- env


def phase_env(args: argparse.Namespace, client: httpx.Client) -> dict[str, Any]:
    smi = run_cmd(
        [
            'nvidia-smi',
            '-i',
            args.gpu,
            '--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits',
        ]
    )
    name, driver, mem_total, mem_used, util = [x.strip() for x in smi.split(',')]
    cuda = re.search(r'CUDA (?:UMD )?Version:\s*([\d.]+)', run_cmd(['nvidia-smi']))
    cpu = next(
        (
            ln.split(':', 1)[1].strip()
            for ln in Path('/proc/cpuinfo').read_text().splitlines()
            if ln.startswith('model name')
        ),
        '',
    )
    meminfo = dict(ln.split(':', 1) for ln in Path('/proc/meminfo').read_text().splitlines())
    names = run_cmd(
        ['docker', 'ps', '--filter', f'name=^{args.project}-', '--format', '{{.Names}}']
    ).split()
    images = dict(
        ln.split(' ', 1)
        for ln in run_cmd(
            ['docker', 'inspect', '--format', '{{.Name}} {{.Config.Image}}', *names]
        ).splitlines()
    )
    trt = run_cmd(
        [
            'docker',
            'exec',
            f'{args.project}-triton',
            'sh',
            '-c',
            'ls /usr/lib/x86_64-linux-gnu | grep -m1 -o "libnvinfer.so.[0-9.]*"',
        ]
    ).strip()
    pin = load_pin(args.manifest)
    before = []
    for _ in range(args.idle_samples):
        out = run_cmd(['nvidia-smi', 'pmon', '-c', '1', '-s', 'u'], timeout=10)
        before.append(parse_pmon(out, args.gpu))
        time.sleep(1.0)
    return {
        'timestamp': datetime.now(UTC).isoformat(),
        'hardware': {
            'gpu': {
                'index': args.gpu,
                'model': name,
                'driver': driver,
                'cuda': cuda[1] if cuda else '',
                'memory_total_mib': float(mem_total),
                'memory_used_before_mib': float(mem_used),
                'utilization_before_pct': float(util),
            },
            'cpu': {'model': cpu, 'logical_cores': os.cpu_count()},
            'ram_gb': int(meminfo['MemTotal'].split()[0]) / 1048576.0,
            'load_average': list(os.getloadavg()),
        },
        'software': {
            'harness_git_sha': run_cmd(['git', '-C', str(_REPO_ROOT), 'rev-parse', 'HEAD']).strip(),
            'stack_version': client.get(f'{args.api_url}/health').json().get('version'),
            'triton_server': client.get(f'{args.triton_url}/v2').json().get('version'),
            'tensorrt_runtime_lib': trt,
            'opensearch': client.get(args.opensearch_url).json().get('version', {}).get('number'),
            'containers': images,
        },
        'dataset': {
            'manifest_sha256': pin['manifest_sha256'],
            'count': pin['count'],
            'seed': pin['seed'],
            'bytes': sum(r['bytes'] for r in pin['images']),
        },
        'foreign_gpu_before': foreign_gpu_summary(before, own_pids(args.project)),
    }
