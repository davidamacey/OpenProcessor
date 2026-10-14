"""Machine probes for the Wave 0 suite: host CPU share, GPU clocks and power, per-GPU
attribution, Triton model reconfiguration, and py-spy sampling of the API container.

The arithmetic is in ``wave0_lib.py``; these classes only read ``/proc``, ``nvidia-smi``,
``docker`` and Triton's HTTP control API.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import httpx

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ruff: noqa: E402
from scripts.bench.suite_lib import foreign_gpu_summary
from scripts.bench.suite_probes import Gauges, PmonSampler, own_pids, run_cmd
from scripts.bench.wave0_lib import (
    clock_summary,
    host_busy_pct,
    parse_clock_line,
    parse_proc_stat,
    rewrite_batching,
    rewrite_instance_group,
)
from src.config.settings import TritonModelConfig


class ClockSampler:
    """SM clock, power and throttle state of every GPU once a second."""

    def __init__(self, interval: float = 1.0) -> None:
        self.interval = interval
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        query = 'index,clocks.sm,power.draw,clocks_throttle_reasons.active'
        while not self._stop.is_set():
            out = run_cmd(
                ['nvidia-smi', f'--query-gpu={query}', '--format=csv,noheader,nounits'], timeout=10
            )
            self.samples.extend(s for s in map(parse_clock_line, out.splitlines()) if s)
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def by_gpu(self) -> dict[str, dict[str, float]]:
        gpus = sorted({int(s['gpu']) for s in self.samples})
        return {
            str(g): clock_summary([s for s in self.samples if int(s['gpu']) == g]) for g in gpus
        }


class Wave0Gauges(Gauges):
    """:class:`Gauges` for one or several GPUs (``gpu`` is a comma list), plus clocks and host CPU.

    ``result()`` keeps the baseline keys (``gpu``, ``attribution`` for the first GPU) and adds
    ``attribution_by_gpu``, ``clocks_by_gpu`` and ``host_busy_pct``.
    """

    def __init__(self, project: str, gpu: str) -> None:
        super().__init__(project, gpu.split(',', maxsplit=1)[0])
        self.all_gpus = gpu.split(',')
        self.pmons = {g: PmonSampler(g) for g in self.all_gpus}
        self.clocks = ClockSampler()
        self._stat0 = (0, 0)
        self._stat1 = (0, 0)

    def __enter__(self) -> Wave0Gauges:
        self.gpus.start()
        for p in self.pmons.values():
            p.start()
        self.clocks.start()
        self._stat0 = parse_proc_stat(Path('/proc/stat').read_text())
        return self

    def __exit__(self, *exc: object) -> None:
        self._stat1 = parse_proc_stat(Path('/proc/stat').read_text())
        self.gpus.stop()
        for p in self.pmons.values():
            p.stop()
        self.clocks.stop()

    def result(self) -> dict[str, Any]:
        own = own_pids(self.project)
        by_gpu = {g: foreign_gpu_summary(p.samples, own) for g, p in self.pmons.items()}
        return {
            'gpu': self.gpus.summary(),
            'attribution': by_gpu[self.all_gpus[0]],
            'attribution_by_gpu': by_gpu,
            'clocks_by_gpu': self.clocks.by_gpu(),
            'host_busy_pct': host_busy_pct(self._stat0, self._stat1),
            'load_average': list(os.getloadavg()),
        }


# ---------------------------------------------------------------- Triton reconfiguration

MODEL_KEYS = {
    'pe': TritonModelConfig.PE_IMAGE_MODEL,
    'det': TritonModelConfig.YOLO_MODEL,
    'face': TritonModelConfig.FACE_DETECT_MODEL,
    'arc': TritonModelConfig.ARCFACE_MODEL,
    'clip': TritonModelConfig.CLIP_IMAGE_MODEL,
    'ocr': TritonModelConfig.OCR_PIPELINE_MODEL,
    'ocrdet': TritonModelConfig.OCR_DET_MODEL,
    'ocrrec': TritonModelConfig.OCR_REC_MODEL,
}


class TritonModels:
    """Edits model configs on the host (``models_dir`` is the stack's mounted model store) and
    reloads the model through Triton's explicit-model-control API; restores the originals."""

    def __init__(self, client: httpx.Client, triton_url: str, models_dir: Path) -> None:
        self.client, self.url, self.dir = client, triton_url, models_dir
        self.originals: dict[str, str] = {}

    def _path(self, model: str) -> Path:
        return self.dir / model / 'config.pbtxt'

    def reload(self, model: str) -> None:
        self.client.post(f'{self.url}/v2/repository/models/{model}/unload').raise_for_status()
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:  # unload is asynchronous
            ready = self.client.get(f'{self.url}/v2/models/{model}/ready').status_code == 200
            if not ready:
                break
            time.sleep(1)
        self.client.post(
            f'{self.url}/v2/repository/models/{model}/load', timeout=600
        ).raise_for_status()
        if self.client.get(f'{self.url}/v2/models/{model}/ready').status_code != 200:
            msg = f'{model} not ready after reload'
            raise RuntimeError(msg)

    def apply(
        self,
        model: str,
        *,
        count: int | None = None,
        gpus: list[int] | None = None,
        queue_delay_us: int | None = None,
        drop_preferred: bool = False,
    ) -> None:
        path = self._path(model)
        text = path.read_text()
        self.originals.setdefault(model, text)
        path.write_text(
            rewrite_batching(
                rewrite_instance_group(text, count=count, gpus=gpus),
                queue_delay_us=queue_delay_us,
                drop_preferred=drop_preferred,
            )
        )
        self.reload(model)

    def restore(self) -> None:
        for model, text in self.originals.items():
            self._path(model).write_text(text)
            self.reload(model)
        self.originals.clear()


# ---------------------------------------------------------------- py-spy


PYSPY_LOOP = (
    'for d in /proc/[0-9]*; do pid=${d#/proc/}; '
    'ppid=$(awk "{print \\$4}" "$d/stat" 2>/dev/null); '
    'if [ "$ppid" = 1 ]; then '
    '/ps/bin/py-spy record --pid "$pid" --duration @SECONDS@ --rate @RATE@ '
    '--format raw --output "/out/@NAME@.$pid.raw" >/dev/null 2>&1 & fi; done; wait'
)


def pyspy_record(
    container: str,
    image: str,
    pyspy_dir: Path,
    out_dir: Path,
    name: str,
    seconds: int,
    rate: int,
) -> list[Path]:
    """Record every worker of the container with a py-spy binary mounted from ``pyspy_dir``.

    Runs a throw-away root container (tracing another user's process needs it) that joins the
    API container's pid namespace and starts one py-spy per child of the container's pid 1 (the
    uvicorn workers; ``--subprocesses`` follows only one of them). Idle threads are not sampled,
    so the samples are proportional to the Python-side time on CPU or holding the GIL.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob(f'{name}.*.raw'):
        old.unlink()
    run_cmd(
        [
            'docker', 'run', '--rm', '--name', f'{container}-pyspy',
            f'--pid=container:{container}', '--cap-add=SYS_PTRACE', '--user', '0',
            '-v', f'{pyspy_dir}:/ps:ro', '-v', f'{out_dir}:/out',
            '--entrypoint', 'sh', image, '-c',
            PYSPY_LOOP.replace('@SECONDS@', str(seconds))
            .replace('@RATE@', str(rate))
            .replace('@NAME@', name),
        ],
        timeout=seconds + 180,
    )  # fmt: skip
    return sorted(out_dir.glob(f'{name}.*.raw'))


THREAD_CENSUS = """
import json, os, sys, time
def snap():
    threads, procs = {}, {}
    for pid in os.listdir('/proc'):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            with open(f'/proc/{pid}/stat') as f:
                ps = f.read()
            rest = ps[ps.rindex(')') + 2:].split()
            comm0 = ps[ps.index('(') + 1:ps.rindex(')')]
            if comm0 == 'py-spy':
                continue
            procs[pid] = (int(rest[11]), int(rest[12]))
            for tid in os.listdir(f'/proc/{pid}/task'):
                with open(f'/proc/{pid}/task/{tid}/stat') as f:
                    s = f.read()
                comm = s[s.index('(') + 1:s.rindex(')')]
                t = s[s.rindex(')') + 2:].split()
                name = comm + (' (main thread)' if tid == pid else '')
                threads[(pid, tid)] = (name, int(t[11]), int(t[12]))
        except OSError:
            continue
    return threads, procs
ta, pa = snap()
time.sleep(float(sys.argv[1]))
tb, pb = snap()
tck = os.sysconf('SC_CLK_TCK')
agg = {}
for key, (name, u1, s1) in tb.items():
    if key not in ta:
        continue
    _, u0, s0 = ta[key]
    row = agg.setdefault(name, [0.0, 0.0, 0])
    row[0] += (u1 - u0) / tck
    row[1] += (s1 - s0) / tck
    row[2] += 1
proc = {'user_s': 0.0, 'system_s': 0.0, 'count': 0, 'new_user_s': 0.0, 'new_system_s': 0.0, 'new_count': 0}
for pid, (u1, s1) in pb.items():
    if pid in pa:
        proc['user_s'] += (u1 - pa[pid][0]) / tck
        proc['system_s'] += (s1 - pa[pid][1]) / tck
        proc['count'] += 1
    else:
        proc['new_user_s'] += u1 / tck
        proc['new_system_s'] += s1 / tck
        proc['new_count'] += 1
print(json.dumps({'threads': agg, 'processes': proc}))
"""


def thread_cpu_census(container: str, seconds: float) -> dict[str, Any]:
    """CPU seconds over a window for every process and thread in the container.

    ``threads`` is keyed by thread name (threads that lived through the whole window), with the
    threads py-spy cannot see (OpenCV, gRPC and BLAS pools, the event loop's native side).
    ``processes`` holds the process-level totals, which also count threads that came and went
    inside the window, and the CPU of processes that started during it.
    """
    proc = subprocess.run(  # nosec B603 B607  fixed argv list, no shell; docker from PATH
        ['docker', 'exec', '-i', container, 'python3', '-', str(seconds)],
        input=THREAD_CENSUS,
        capture_output=True,
        text=True,
        timeout=seconds + 120,
        check=True,
    )
    raw = json.loads(proc.stdout.strip().splitlines()[-1])
    return {
        'threads': {
            k: {'user_s': v[0], 'system_s': v[1], 'threads': v[2]}
            for k, v in raw['threads'].items()
        },
        'processes': raw['processes'],
        'seconds': seconds,
    }
