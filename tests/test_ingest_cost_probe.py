"""Arithmetic of the ingest cost probe (the numbers quoted in docs/PERFORMANCE.md)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'bench' / 'ingest_cost_probe.py'
_spec = importlib.util.spec_from_file_location('ingest_cost_probe', SCRIPT)
assert _spec is not None
assert _spec.loader is not None
probe = importlib.util.module_from_spec(_spec)
sys.modules['ingest_cost_probe'] = probe
_spec.loader.exec_module(probe)

KW = {
    'n_images': 1000,
    'metadata_bytes': 1800.0,
    'vector_bytes': 8400.0,
    'frame_vector_bytes': 8400.0,
}


def _row(items: float, frac: float):
    return probe.cost_row(probe.Scenario('s', items, frac), **KW)


def test_narrow_embed_all_matches_plan_table() -> None:
    row = _row(1.37, 1.0)
    assert row.item_store_mb == pytest.approx(14.0, abs=0.1)
    assert row.total_mb == pytest.approx(22.4, abs=0.1)


def test_full_vocabulary_ratio_and_forwards() -> None:
    rows = [_row(1.37, 1.0), _row(7.0, 1.0), _row(7.0, 0.0)]
    r = probe.ratios(rows)
    assert r[1] == pytest.approx(3.6, abs=0.05)
    assert r[2] == pytest.approx(0.94, abs=0.02)
    assert rows[1].crop_forwards == 7000
    assert rows[2].crop_forwards == 0


def test_rejects_out_of_range_fraction() -> None:
    with pytest.raises(ValueError, match='embedded_fraction'):
        _row(1.0, 1.5)
