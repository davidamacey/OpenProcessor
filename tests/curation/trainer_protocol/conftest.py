"""Fixtures for the trainer conformance tests: the shared jobs volume and a frozen export."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


TRAINER_DIR = Path(__file__).resolve().parents[3] / 'docker' / 'trainer'
if str(TRAINER_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINER_DIR))


@pytest.fixture
def jobs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A shared /jobs volume both halves of the protocol point at.

    Also points ``default``'s ``export_root`` (and so its stamped
    ``project_export_root``, projects_plan.md §5.3) at ``tmp_path``'s
    project data root via ``OP_PROJECTS_DATA_ROOT`` (``OP_EXPORT_ROOT``
    is retired -- P1 removed the env-derived default special case) --
    every fixture's ``export_dir`` lives under it, so the trainer's
    export-containment check never rejects these fixtures' export dirs
    as escaping the project. Resets the cached ``CurationConfig``
    singleton (the codebase's existing pattern, e.g.
    ``test_export_datasets.py``) so the env var actually takes. Returns
    the resolved ``.../projects/default`` subdir -- default is an
    ordinary project (no special-casing), so that's where job files
    actually land.
    """
    import src.config.curation_accessors as curation_config_mod

    d = tmp_path / 'jobs'
    d.mkdir()
    monkeypatch.setenv('OP_TRAIN_JOBS_DIR', str(d))
    # default's export_root resolves to OP_PROJECTS_DATA_ROOT/default/exports
    # (P1: no special-casing); point OP_PROJECTS_DATA_ROOT at tmp_path
    # directly so export_dir (below) can build under exactly that path.
    monkeypatch.setenv('OP_PROJECTS_DATA_ROOT', str(tmp_path))
    monkeypatch.setattr(curation_config_mod, '_default_curation_config', None)
    resolved = d / 'projects' / 'default'
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


@pytest.fixture
def export_dir(tmp_path: Path, jobs_dir: Path) -> Path:
    """A minimal frozen export: 3 classes, one labeled image per split.

    Depends on ``jobs_dir`` for its ``OP_PROJECTS_DATA_ROOT`` side
    effect (must run first) and builds under ``default``'s real
    ``export_root`` (``OP_PROJECTS_DATA_ROOT/default/exports``) so the
    trainer's export-containment check never rejects it.
    """
    root = tmp_path / 'default' / 'exports' / 'export'
    for split in ('train', 'val', 'test'):
        (root / 'images' / split).mkdir(parents=True)
        (root / 'labels' / split).mkdir(parents=True)
        (root / 'images' / split / f'{split}0.jpg').write_bytes(b'not-a-real-jpeg')
        # Dense export ids 0/1/2 -> registry ids 5/9/12 (see export_id_map).
        (root / 'labels' / split / f'{split}0.txt').write_text(
            '0 0.5 0.5 0.2 0.2\n1 0.25 0.25 0.1 0.1\n2 0.75 0.75 0.1 0.1\n'
        )
    (root / 'data.yaml').write_text(
        'path: .\n'
        'train: images/train\n'
        'val: images/val\n'
        'test: images/test\n'
        'nc: 3\n'
        "names: ['widget', 'serial_plate', 'gadget']\n"
    )
    # A registry id space that is deliberately NOT the dense export id space:
    # this is exactly the translation subset_dataset must not skip.
    (root / 'class_registry.json').write_text(
        json.dumps(
            {
                'classes': [
                    {'class_id': 5, 'class_name': 'widget'},
                    {'class_id': 9, 'class_name': 'serial_plate'},
                    {'class_id': 12, 'class_name': 'gadget'},
                ],
                'export_id_map': {'5': 0, '9': 1, '12': 2},
            }
        )
    )
    (root / 'manifest.json').write_text(json.dumps({'frozen_test_sha': 'sha-frozen-test'}))
    return root
