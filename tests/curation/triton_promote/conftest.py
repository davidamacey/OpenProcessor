"""Fixtures for the Triton promoter tests: a scratch model repo, a fake checkpoint and a warm-up stub."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from src.services.training.job_models import TrainJobStatus
from src.services.training.triton_promote import TritonPromoter


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def fake_checkpoint(tmp_path: Path) -> Path:
    """A fake run dir with best.pt + best.onnx, like the trainer produces."""
    run_dir = tmp_path / 'jobs' / 'job-abc'
    run_dir.mkdir(parents=True)
    (run_dir / 'best.pt').write_bytes(b'not-a-real-checkpoint')
    (run_dir / 'best.onnx').write_bytes(b'ONNX-V1-BYTES')
    return run_dir / 'best.pt'


@pytest.fixture
def fake_status(fake_checkpoint: Path) -> TrainJobStatus:
    return TrainJobStatus(
        job_id='job-abc',
        state='finished',
        checkpoint_path=str(fake_checkpoint),
        eval={'map50': 0.9, 'per_class': []},
    )


@pytest.fixture
def scratch_models_dir(tmp_path: Path) -> Path:
    """Disposable stand-in for the Triton model repo — never the real one."""
    d = tmp_path / 'scratch_triton_models'
    d.mkdir()
    return d


@pytest.fixture(autouse=True)
def _no_real_warm_up(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """promote() warms the model up over HTTP; tests that are not about that
    must never reach a real Triton URL."""
    if 'warm_up' in request.node.name:
        return
    monkeypatch.setattr(TritonPromoter, '_warm_up', AsyncMock(return_value=False))
