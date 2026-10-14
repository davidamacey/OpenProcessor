"""Host and container observations recorded with every result (the host is shared with other work)."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


PROJECT = 'vecdb'


def cpu_times() -> tuple[int, int]:
    with Path('/proc/stat').open() as fh:
        fields = [int(x) for x in fh.readline().split()[1:]]
    idle = fields[3] + fields[4]
    return sum(fields), idle


def host_snapshot(window_s: float = 1.0) -> dict[str, Any]:
    """Load averages and whole-host CPU busy percent over a short window."""
    t0, i0 = cpu_times()
    time.sleep(window_s)
    t1, i1 = cpu_times()
    busy = 100.0 * (1 - (i1 - i0) / max(1, t1 - t0))
    load = os.getloadavg()
    return {
        'load1': load[0],
        'load5': load[1],
        'load15': load[2],
        'host_cpu_busy_pct': round(busy, 1),
        'cores': os.cpu_count(),
    }


def container_name(service: str) -> str:
    return f'{PROJECT}-{service}-1'


def _docker(*args: str) -> str:
    out = subprocess.run(['docker', *args], capture_output=True, text=True, check=True, timeout=300)  # nosec B603 B607  fixed argv list, no shell; docker from PATH
    return out.stdout


def du_bytes(service: str, path: str) -> int:
    out = _docker('exec', container_name(service), 'du', '-sb', path)
    return int(out.split()[0])


def cgroup_memory(service: str) -> dict[str, int]:
    """Anonymous and file-backed memory of the container's cgroup, in bytes."""
    text = _docker('exec', container_name(service), 'cat', '/sys/fs/cgroup/memory.stat')
    stat = {line.split()[0]: int(line.split()[1]) for line in text.splitlines()}
    current = int(_docker('exec', container_name(service), 'cat', '/sys/fs/cgroup/memory.current'))
    return {
        'anon': stat.get('anon', 0),
        'file': stat.get('file', 0),
        'shmem': stat.get('shmem', 0),
        'current': current,
    }


def cpu_seconds(service: str) -> float:
    """Cumulative CPU time of the container's cgroup, in seconds."""
    text = _docker('exec', container_name(service), 'cat', '/sys/fs/cgroup/cpu.stat')
    for line in text.splitlines():
        if line.startswith('usage_usec'):
            return int(line.split()[1]) / 1e6
    return float('nan')


def image_digest(service: str) -> dict[str, str]:
    inspect = json.loads(_docker('inspect', container_name(service)))[0]
    image = inspect['Config']['Image']
    info = json.loads(_docker('image', 'inspect', image))[0]
    return {'image': image, 'id': info['Id'], 'repo_digests': ','.join(info.get('RepoDigests', []))}
