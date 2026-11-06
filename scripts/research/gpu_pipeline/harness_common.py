"""Shared plumbing of the GPU pipeline prototypes: image sets, environment capture, NVML sampling,
statistics. No dataset path or GPU index has a default that points at a private location."""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
PREP_DIR = REPO_ROOT / 'models' / 'gpu_image_prep' / '1'
DEFAULT_SEED = 20261104


def use_shipped_prep_modules() -> None:
    """Make the shipped ``gpu_image_prep`` modules importable by their flat names."""
    for path in (str(REPO_ROOT), str(PREP_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)


# --------------------------------------------------------------------------------- image sets


def ordered_files(
    root: Path, *, pin: Path | None = None, manifest: Path | None = None
) -> list[Path]:
    """The files of one set in the registered order.

    ``pin``: a bench pin JSON (``images[*].file_name`` in pin order, resolved under ``root`` or
    ``root/images``). ``manifest``: the private sample manifest (``files[*].name``, draw order).
    Neither: the sorted ``*.jpg`` names of ``root``.
    """
    if pin is not None:
        rows = json.loads(pin.read_text())['images']
        # pins name files either explicitly or by image id
        names = [
            row['file_name'] if 'file_name' in row else f'{row["image_id"]}.jpg' for row in rows
        ]
    elif manifest is not None:
        names = [row['name'] for row in json.loads(manifest.read_text())['files']]
    else:
        names = sorted(p.name for p in root.glob('*.jpg'))
    base = root / 'images' if (root / 'images').is_dir() else root
    return [base / n for n in names]


def take(files: list[Path], n: int) -> list[Path]:
    """The first ``n`` files; a set shorter than ``n`` is an error, never a silent shrink."""
    if len(files) < n:
        raise ValueError(f'set has {len(files)} files, registered sample is {n}')
    return files[:n]


def drop_page_cache(paths: list[Path]) -> None:
    """``posix_fadvise(DONTNEED)`` per file so a repetition reads from disk, as 07 M-4 asks."""
    for p in paths:
        fd = os.open(p, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


# --------------------------------------------------------------------------------- environment


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(  # nosec B603
            cmd, capture_output=True, text=True, timeout=30, check=False
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ''


def environment(gpu_index: int | None) -> dict[str, Any]:
    """Versions, driver, GPU identity and host load: recorded beside every result."""
    env: dict[str, Any] = {
        'python': platform.python_version(),
        'host_load_1m': os.getloadavg()[0],
        'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'commit': _run(['git', '-C', str(REPO_ROOT), 'rev-parse', 'HEAD']),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'packages': {d.metadata['Name']: d.version for d in metadata.distributions()},
    }
    if gpu_index is not None:
        q = _run(
            [
                'nvidia-smi',
                f'--id={gpu_index}',
                '--query-gpu=name,uuid,driver_version,clocks.max.sm,power.limit,memory.total',
                '--format=csv,noheader',
            ]
        )
        env['gpu'] = q
        env['compute_apps'] = _run(
            [
                'nvidia-smi',
                f'--id={gpu_index}',
                '--query-compute-apps=pid,name',
                '--format=csv,noheader',
            ]
        )
    return env


@dataclass
class NvmlSampler:
    """Samples utilization, clocks, power and memory of one GPU every ``period`` seconds."""

    index: int
    period: float = 0.1
    rows: list[dict[str, float]] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def __enter__(self) -> NvmlSampler:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def _loop(self) -> None:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(self.index)
        while not self._stop.is_set():
            u = pynvml.nvmlDeviceGetUtilizationRates(h)
            self.rows.append(
                {
                    't': time.perf_counter(),
                    'gpu_util': float(u.gpu),
                    'mem_util': float(u.memory),
                    'sm_clock': float(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)),
                    'power_w': pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0,
                    'mem_mib': pynvml.nvmlDeviceGetMemoryInfo(h).used / 1048576.0,
                }
            )
            self._stop.wait(self.period)
        pynvml.nvmlShutdown()

    def summary(self) -> dict[str, float]:
        if not self.rows:
            return {'samples': 0}
        out: dict[str, float] = {'samples': float(len(self.rows))}
        for key in ('gpu_util', 'sm_clock', 'power_w', 'mem_mib'):
            out[f'{key}_mean'] = float(np.mean([r[key] for r in self.rows]))
        out['mem_mib_peak'] = float(max(r['mem_mib'] for r in self.rows))
        return out


# --------------------------------------------------------------------------------- statistics


def spread(values: list[float]) -> float:
    """(max - min) / median, the registered spread of repetitions."""
    med = float(np.median(values))
    return (max(values) - min(values)) / med if med else math.inf


def paired_bootstrap_ratio(
    cand: list[float], ctrl: list[float], *, draws: int = 5000, seed: int = DEFAULT_SEED
) -> dict[str, float]:
    """Median of cand / ctrl over paired samples with a 95 % bootstrap interval over pairs."""
    a, b = np.asarray(cand, float), np.asarray(ctrl, float)
    if a.shape != b.shape or a.size == 0:
        raise ValueError('paired samples must be non-empty and equal in length')
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(draws, a.size))
    boot = np.median(a[idx], axis=1) / np.median(b[idx], axis=1)
    return {
        'ratio': float(np.median(a) / np.median(b)),
        'lo': float(np.percentile(boot, 2.5)),
        'hi': float(np.percentile(boot, 97.5)),
        'pairs': float(a.size),
    }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=float))
