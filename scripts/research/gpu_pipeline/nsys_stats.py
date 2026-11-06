"""Run a command under ``nsys`` and read back GPU kernel time per NVTX range and the busy share.

``nsys profile --trace=cuda,nvtx`` then ``nsys stats -r nvtx_kern_sum`` (kernel time attributed to
the NVTX range that launched it) and a SQLite read of the CUDA kernel intervals for the share of
wall time with at least one kernel running. GPU performance counters (SM Active) need profiling
rights this host does not grant to users (``ERR_NVGPUCTRPERM``); this is the substitute.
"""

from __future__ import annotations

import csv
import io
import sqlite3
import subprocess
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pathlib import Path


def profile(command: list[str], out: Path) -> Path:
    """Profile ``command``; returns the ``.sqlite`` export. The caller wraps this in the card lock."""
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(  # nosec B603 B607
        [
            'nsys',
            'profile',
            '-o',
            str(out),
            '-f',
            'true',
            '--trace=cuda,nvtx',
            '--export=sqlite',
            *command,
        ],
        check=True,
        capture_output=True,
    )
    return out.with_suffix('.sqlite')


def nvtx_kernel_ms(report: Path) -> dict[str, float]:
    """Total kernel milliseconds per NVTX range name (``:name`` push/pop ranges)."""
    text = subprocess.run(  # nosec B603 B607
        ['nsys', 'stats', '-r', 'nvtx_kern_sum', '-f', 'csv', '--force-export=true', str(report)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    body = text[text.index('NVTX Range,') :]
    totals: dict[str, float] = {}
    for row in csv.DictReader(io.StringIO(body)):
        name = row['NVTX Range'].lstrip(':')
        totals[name] = totals.get(name, 0.0) + float(row['Total Time (ns)']) / 1e6
    return totals


def kernel_busy(sqlite_path: Path) -> dict[str, float]:
    """Total kernel and memcpy ms, the kernel window in ms and the share of it with a kernel running."""
    con = sqlite3.connect(sqlite_path)
    rows = con.execute(
        'SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start'
    ).fetchall()
    has_copies = con.execute(
        "SELECT 1 FROM sqlite_master WHERE name = 'CUPTI_ACTIVITY_KIND_MEMCPY'"
    ).fetchone()
    copy_ns = (
        con.execute(
            'SELECT COALESCE(SUM(end - start), 0) FROM CUPTI_ACTIVITY_KIND_MEMCPY'
        ).fetchone()[0]
        if has_copies
        else 0
    )
    con.close()
    if not rows:
        return {'kernel_ms': 0.0, 'memcpy_ms': copy_ns / 1e6, 'window_ms': 0.0, 'busy_share': 0.0}
    union, cur_s, cur_e = 0, rows[0][0], rows[0][1]
    for s, e in rows[1:]:
        if s > cur_e:
            union += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    union += cur_e - cur_s
    window = rows[-1][1] - rows[0][0]
    return {
        'kernel_ms': sum(e - s for s, e in rows) / 1e6,
        'memcpy_ms': copy_ns / 1e6,
        'window_ms': window / 1e6,
        'busy_share': union / window if window else 0.0,
    }
