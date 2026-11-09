"""A cancelled bake-off stops submitting (dataset, model) pairs and ends ``cancelled``.

The runner is driven with a fake scoring backend in place of the per-pair subprocess; only the
pair loop's cancel check is under test.
"""

from __future__ import annotations

import json
import threading
from typing import TYPE_CHECKING, Any

import pytest

from scripts.curation.bakeoff import bakeoff_runner


if TYPE_CHECKING:
    from pathlib import Path


PAIRS = 6


@pytest.fixture
def scored(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The models the fake backend scored, in start order."""
    started: list[str] = []
    lock = threading.Lock()

    def fake_task(_ds_path, _ds_out, _ds_id, model, _gpu):
        with lock:
            started.append(model['model'])
        return model['model'], True, None

    monkeypatch.setattr(bakeoff_runner, '_run_task', fake_task)
    monkeypatch.setattr(bakeoff_runner, '_CONCURRENCY', 2)
    monkeypatch.setattr(bakeoff_runner, 'test_sha', lambda _path: ('sha', 0))
    return started


def _spec(tmp_path: Path) -> dict[str, Any]:
    return {
        'schema_version': bakeoff_runner.SCHEMA_VERSION,
        'job_id': 'j1',
        'out_dir': str(tmp_path / 'out'),
        'datasets': [
            {'id': 'd1', 'dir_name': 'd1', 'path': str(tmp_path), 'test_label_sha': 'sha'}
        ],
        'models': [{'model': f'm{i}'} for i in range(PAIRS)],
    }


def test_a_cancel_between_pairs_stops_submitting(scored: list[str], tmp_path: Path) -> None:
    status = bakeoff_runner.run_job(_spec(tmp_path), cancelled=lambda: bool(scored))

    assert status['state'] == 'cancelled'
    assert 1 <= len(scored) < PAIRS
    assert len(status['completed']) == len(scored)
    assert status['progress']['done'] == len(scored)
    written = json.loads((tmp_path / 'out' / 'status.json').read_text())
    assert written['state'] == 'cancelled'
    assert not (tmp_path / 'out' / 'matrix.json').exists()  # no matrix over a partial grid


def test_without_a_cancel_every_pair_is_scored(scored: list[str], tmp_path: Path) -> None:
    status = bakeoff_runner.run_job(_spec(tmp_path), cancelled=lambda: False)

    assert status['state'] == 'done'
    assert sorted(scored) == sorted(f'm{i}' for i in range(PAIRS))
