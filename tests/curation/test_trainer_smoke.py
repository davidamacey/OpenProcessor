"""End-to-end smoke test: a *real* Ultralytics training run through the trainer.

the ``trainer_protocol`` tests stub Ultralytics out, which proves the trainer's
orchestration but not that it can actually drive a training loop. This
module runs the genuine thing -- ``trainer.run_training`` -> ``ultralytics`` ->
``best.pt`` -> ``results.csv`` -> the run state and manifest -- on a
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

import json
import os
import shutil
import sys
from pathlib import Path

import pytest


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

    import job_spec
    import job_status
    import trainer
    import trainer_kwargs
    from dataset_prep import read_class_remap
    from job_manifest import build_manifest

    from curation.trainer_protocol._helpers import make_spec

    monkeypatch.setattr(trainer_kwargs, 'RUNS_ROOT', str(tmp_path / 'runs'))
    monkeypatch.setattr(job_spec, 'TMP_ROOT', tmp_path / 'scratch')
    # Keep Ultralytics' settings/cache inside tmp_path.
    monkeypatch.setenv('YOLO_CONFIG_DIR', str(tmp_path / 'ultralytics'))
    # No incumbent -> no Triton call from the comparison leg.
    monkeypatch.delenv('OP_TRAIN_INCUMBENT_MODELS', raising=False)
    # File-backed MLflow store: exercises the tracking path with no server.
    monkeypatch.setenv('MLFLOW_TRACKING_URI', f'file://{tmp_path / "mlruns"}')

    export = _build_synthetic_export(tmp_path / 'export')

    spec = make_spec(
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

    state = job_status.StatusState(job_id=spec.job_id, state='starting')
    try:
        trainer.run_training(spec, state)
        manifest = build_manifest(spec, state, class_remap=read_class_remap(spec))
    finally:
        shutil.rmtree(spec.tmp_root, ignore_errors=True)

    assert state.state == 'finished', state.error
    assert state.checkpoint_path is not None
    best_pt = Path(state.checkpoint_path)
    assert best_pt.is_file(), 'Ultralytics produced no best.pt'
    assert best_pt.stat().st_size > 0
    # promote copies this ONNX into the Triton model repo; without it a
    # finished run is unservable.
    assert best_pt.with_suffix('.onnx').is_file(), 'ONNX export did not run'
    # results.csv -> eval block is what the runs list and promote gate read.
    assert state.eval is not None
    assert 'map50' in state.eval

    assert manifest is not None
    assert manifest['results']['final_state'] == 'finished'
    assert manifest['results']['checkpoint_sha256']
    assert manifest['lineage']['dataset_sha'] == 'smoke-frozen-sha'
    assert manifest['lineage']['training_seed'] == 11
    assert manifest['code_versions']['ultralytics_pkg']

    # The per-job scratch dir is always removed, even on success.
    assert not spec.tmp_root.exists()
