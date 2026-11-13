"""Fixtures for the trainer conformance tests: a frozen export."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


TRAINER_DIR = Path(__file__).resolve().parents[3] / 'docker' / 'trainer'
if str(TRAINER_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINER_DIR))


@pytest.fixture
def export_dir(tmp_path: Path) -> Path:
    """A minimal frozen export: 3 classes, one labeled image per split."""
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
