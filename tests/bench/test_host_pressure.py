"""PSI reader: parsed from a fake pressure file, no kernel needed."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.bench import host_pressure


if TYPE_CHECKING:
    from pathlib import Path


def test_pressure_io_returns_cumulative_some_and_full(tmp_path: Path) -> None:
    f = tmp_path / 'io'
    f.write_text(
        'some avg10=0.50 avg60=0.20 avg300=0.10 total=1234567\n'
        'full avg10=0.00 avg60=0.00 avg300=0.00 total=89\n'
    )
    assert host_pressure.pressure_io(f) == (1234567.0, 89.0)
