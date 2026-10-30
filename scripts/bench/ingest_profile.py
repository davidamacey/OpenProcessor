"""Stage-by-stage ingest profile: run a row with the measurement protocol, then print its table.

``run`` wraps ``run_baseline.py`` with the protocol of the ship-track plan: three repetitions
(fresh project each), the input page cache dropped before each, GPU sampled every 500 ms, the load
average recorded at start and end, plus the API container's CPU seconds (cgroup) and the Triton
container's CPU (``docker stats``) over the run. ``table`` renders the per-stage table of rows
already run. Raw JSON goes to ``OUT_ROOT/ROW/``; the table holds aggregates only.

    ingest_profile.py run setB MANIFEST --api-url http://127.0.0.1:5703 \\
        --triton-metrics-url http://127.0.0.1:5702/metrics --path-map HOST=CONTAINER \\
        --api-container NAME --triton-container NAME --out-root artifacts_local/bench/v070/sp0
    ingest_profile.py table setA=artifacts_local/bench/v070/sp0/setA setB=...
"""

from __future__ import annotations

import argparse
import contextlib
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.bench import run_baseline as rb
from scripts.bench.ingest_profile_report import load_reps, render_row


PROTOCOL = {'reps': '3', 'gpu_interval': '0.5', 'policy': 'all', 'stages': 'ingest'}
STATS_EVERY_S = 2.0
CGROUP_CPU_STAT = '/sys/fs/cgroup/cpu.stat'


def docker(*argv: str) -> str:
    return subprocess.run(  # nosec B603 B607  fixed argv list, no shell
        ['docker', *argv], capture_output=True, text=True, check=True, timeout=30
    ).stdout


class ContainerObserver:
    """API CPU seconds (cgroup usage delta) and mean Triton CPU percent (100 = one core)."""

    def __init__(
        self,
        api: str,
        triton: str,
        runner: Callable[..., str] = docker,
        every: float = STATS_EVERY_S,
    ) -> None:
        self._api, self._triton, self._run, self._every = api, triton, runner, every
        self._cpu0 = 0.0
        self._cpu_pct: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _usage_s(self) -> float:
        text = self._run('exec', self._api, 'cat', CGROUP_CPU_STAT)
        match = re.search(r'^usage_usec (\d+)$', text, re.M)
        if match is None:
            raise ValueError(f'{self._api}: no usage_usec in {CGROUP_CPU_STAT}')
        return int(match.group(1)) / 1e6

    def sample_triton(self) -> None:
        out = self._run('stats', '--no-stream', '--format', '{{.CPUPerc}}', self._triton)
        self._cpu_pct.append(float(out.strip().rstrip('%')))

    def _loop(self) -> None:
        while not self._stop.is_set():
            # a missed sample is absent from the mean, never invented
            with contextlib.suppress(subprocess.SubprocessError, OSError, ValueError):
                self.sample_triton()
            self._stop.wait(self._every)

    def start(self) -> None:
        self._cpu0 = self._usage_s()
        self._cpu_pct = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=35)
        out = {'api_cpu_s': self._usage_s() - self._cpu0}
        if self._cpu_pct:
            out['triton_cpu_pct_mean'] = sum(self._cpu_pct) / len(self._cpu_pct)
            out['triton_cpu_samples'] = float(len(self._cpu_pct))
        return out


def build_run_args(ns: argparse.Namespace) -> argparse.Namespace:
    """The ``run_baseline`` arguments of one profile row; the protocol values are fixed here."""
    argv = [
        str(ns.manifest),
        '--api-url', ns.api_url,
        '--slug-prefix', f'sp0-{ns.row}',
        '--policy', PROTOCOL['policy'],
        '--stages', PROTOCOL['stages'],
        '--reps', str(ns.reps),
        '--gpu-interval', PROTOCOL['gpu_interval'],
        '--warmup', str(ns.warmup),
        '--drop-cache',
        '--out', str(ns.out_root / ns.row / ns.row),
    ]  # fmt: skip
    if ns.triton_metrics_url:
        argv += ['--triton-metrics-url', ns.triton_metrics_url]
    for spec in ns.path_map:
        argv += ['--path-map', spec]
    return rb.build_parser().parse_args(argv)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    run = sub.add_parser('run')
    run.add_argument('row', help='row name, e.g. setA, setB, setP3000')
    run.add_argument('manifest', type=Path)
    run.add_argument('--api-url', required=True)
    run.add_argument('--triton-metrics-url')
    run.add_argument('--api-container', required=True)
    run.add_argument('--triton-container', required=True)
    run.add_argument('--path-map', action='append', default=[], metavar='HOST=CONTAINER')
    run.add_argument('--out-root', type=Path, required=True)
    run.add_argument('--reps', type=int, default=int(PROTOCOL['reps']))
    run.add_argument('--warmup', type=int, default=100)
    table = sub.add_parser('table')
    table.add_argument('rows', nargs='+', metavar='NAME=DIR')
    return p


def main(argv: Sequence[str] | None = None) -> int:
    ns = _parser().parse_args(argv)
    if ns.cmd == 'table':
        for spec in ns.rows:
            name, _, directory = spec.partition('=')
            print(render_row(name, load_reps(Path(directory))))
        return 0
    observer = ContainerObserver(ns.api_container, ns.triton_container)
    rb.run_reps(build_run_args(ns), observer=observer)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
