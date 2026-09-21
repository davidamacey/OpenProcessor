"""End-to-end smoke test: a *real* Ultralytics training run through the trainer.

``test_trainer_protocol.py`` stubs Ultralytics out, which proves the file
protocol but not that the trainer can actually drive a training loop. This
module runs the genuine thing -- ``trainer.run_job`` -> ``ultralytics`` ->
``best.pt`` -> ``results.csv`` -> ``status.json`` -> ``manifest.json`` -- on a
synthetic four-image dataset, one epoch, CPU only.

It is **opt-in**: set ``OP_TRAINER_SMOKE=1`` to enable. It needs
``ultralytics``, ``torch`` and ``cv2`` importable in the running environment
and takes tens of seconds, so it stays out of the default gate. It never
touches a GPU, downloads no weights (the model is built from an architecture
YAML, not a pretrained checkpoint), and writes only under ``tmp_path``.

    OP_TRAINER_SMOKE=1 .venv/bin/python -m pytest \\
        tests/curation/test_trainer_smoke.py -v --no-cov
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from src.services.training import jobs as train_jobs
from src.services.training.jobs import TrainJobSpec


TRAINER_DIR = Path(__file__).resolve().parents[2] / 'docker' / 'trainer'
if str(TRAINER_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINER_DIR))

pytestmark = pytest.mark.skipif(
    os.environ.get('OP_TRAINER_SMOKE') != '1',
    reason='real training run; set OP_TRAINER_SMOKE=1 to enable',
)

# Architecture YAML rather than a ``.pt``: Ultralytics builds it from its own
# packaged config with no download, so the smoke test needs no network. Any
# detection architecture the installed Ultralytics ships works here; the point
# is the trainer's orchestration, not the model.
SMOKE_ARCH = os.environ.get('OP_TRAINER_SMOKE_ARCH', 'yolo11n.yaml')

IMG_SIZE = 64
N_TRAIN_IMAGES = 4


def _build_synthetic_export(root: Path) -> Path:
    """Write a tiny single-class YOLO export: solid squares on noise."""
    import cv2
    import numpy as np

    rng = np.random.default_rng(1234)
    counts = {'train': N_TRAIN_IMAGES, 'val': 2, 'test': 2}
    for split, n in counts.items():
        (root / 'images' / split).mkdir(parents=True)
        (root / 'labels' / split).mkdir(parents=True)
        for i in range(n):
            img = rng.integers(0, 60, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            x0, y0 = 16 + (i % 2) * 8, 16 + (i % 3) * 4
            x1, y1 = x0 + 20, y0 + 20
            img[y0:y1, x0:x1] = 235
            cv2.imwrite(str(root / 'images' / split / f'{split}{i}.png'), img)
            cx, cy = (x0 + x1) / 2 / IMG_SIZE, (y0 + y1) / 2 / IMG_SIZE
            w, h = (x1 - x0) / IMG_SIZE, (y1 - y0) / IMG_SIZE
            (root / 'labels' / split / f'{split}{i}.txt').write_text(
                f'0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n'
            )
    (root / 'data.yaml').write_text(
        f'path: {root}\n'
        'train: images/train\n'
        'val: images/val\n'
        'test: images/test\n'
        'nc: 1\n'
        "names: ['block']\n"
    )
    (root / 'manifest.json').write_text(json.dumps({'frozen_test_sha': 'smoke-frozen-sha'}))
    return root


@pytest.mark.integration
def test_a_real_training_run_completes_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip('ultralytics')
    pytest.importorskip('torch')
    pytest.importorskip('cv2')

    import job_protocol
    import trainer

    jobs_dir = tmp_path / 'jobs'
    jobs_dir.mkdir()
    monkeypatch.setenv('OP_TRAIN_JOBS_DIR', str(jobs_dir))
    monkeypatch.setattr(trainer, 'RUNS_ROOT', str(tmp_path / 'runs'))
    monkeypatch.setattr(job_protocol, 'TMP_ROOT', tmp_path / 'scratch')
    # Keep Ultralytics' settings/cache inside tmp_path.
    monkeypatch.setenv('YOLO_CONFIG_DIR', str(tmp_path / 'ultralytics'))
    # No incumbent -> no Triton call from the comparison leg.
    monkeypatch.delenv('OP_TRAIN_INCUMBENT_MODELS', raising=False)
    # File-backed MLflow store: exercises the tracking path with no server.
    monkeypatch.setenv('MLFLOW_TRACKING_URI', f'file://{tmp_path / "mlruns"}')

    export = _build_synthetic_export(tmp_path / 'export')

    job_id = asyncio.run(
        train_jobs.write_job(
            TrainJobSpec(
                dataset_export_dir=str(export),
                model_size='n',
                profile='probe',
                hyperparameters={
                    'model': SMOKE_ARCH,
                    'epochs': 1,
                    'batch': 2,
                    'imgsz': IMG_SIZE,
                    # MuSGD is the pipeline default but may not exist in every
                    # installed Ultralytics; SGD always does and the optimizer
                    # choice is irrelevant to what this test proves.
                    'optimizer': 'SGD',
                    'device': 'cpu',
                    'workers': 0,
                    'plots': False,
                    'val': True,
                    'verbose': False,
                    'seed': 11,
                },
            )
        )
    )
    spec = job_protocol.parse_and_validate_job(jobs_dir / f'{job_id}.job.json')

    trainer.run_job(spec)

    status = asyncio.run(train_jobs.read_status(job_id))
    assert status is not None
    assert status.state == 'finished', status.error
    assert status.checkpoint_path is not None
    best_pt = Path(status.checkpoint_path)
    assert best_pt.is_file(), 'Ultralytics produced no best.pt'
    assert best_pt.stat().st_size > 0
    # promote copies this ONNX into the Triton model repo; without it a
    # finished run is unservable.
    assert best_pt.with_suffix('.onnx').is_file(), 'ONNX export did not run'
    # results.csv -> eval block is what the runs list and promote gate read.
    assert status.eval is not None
    assert 'map50' in status.eval

    manifest = asyncio.run(train_jobs.read_manifest(job_id))
    assert manifest is not None
    assert manifest['results']['final_state'] == 'finished'
    assert manifest['results']['checkpoint_sha256']
    assert manifest['lineage']['dataset_sha'] == 'smoke-frozen-sha'
    assert manifest['lineage']['training_seed'] == 11
    assert manifest['code_versions']['ultralytics_pkg']

    # The per-job scratch dir is always removed, even on success.
    assert not spec.tmp_root.exists()
