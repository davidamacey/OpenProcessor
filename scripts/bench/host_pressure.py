"""Host IO pressure (PSI) reader shared by the bench drivers."""

from __future__ import annotations

from pathlib import Path


PRESSURE_IO = Path('/proc/pressure/io')


def pressure_io(path: Path = PRESSURE_IO) -> tuple[float, float]:
    """Cumulative (some, full) IO stall microseconds of the host (/proc/pressure/io)."""
    vals = {}
    for line in path.read_text().splitlines():
        kind, *rest = line.split()
        vals[kind] = float(dict(kv.split('=') for kv in rest)['total'])
    return vals['some'], vals['full']
