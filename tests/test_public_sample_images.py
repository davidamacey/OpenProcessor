"""Sample images in tests, demos and docs are public COCO / Open Images files (#445).

The Ultralytics demo images (a bus and a footballer) were fetched by the Makefile, the setup
smoke test and the model validation; their ground truth and licensing notes lived in the tree.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
# Built from parts so this file does not match itself.
FORBIDDEN = ('zid' + 'ane', 'ultralytics.com/' + 'images', 'bus' + '.jpg')
# Release notes name what was replaced.
EXEMPT = {'CHANGELOG.md', 'frontend/CHANGELOG.md'}


def _tracked_text_files() -> list[Path]:
    out = subprocess.run(
        ['git', 'ls-files', '-z'], cwd=ROOT, capture_output=True, check=True
    ).stdout.decode()
    return [ROOT / n for n in out.split('\0') if n and n not in EXEMPT]


def test_no_tracked_file_names_the_ultralytics_sample_images() -> None:
    hits: list[str] = []
    for path in _tracked_text_files():
        if not path.is_file() or path.suffix in {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.ico'}:
            continue
        try:
            text = path.read_text(encoding='utf-8')
        except (UnicodeDecodeError, OSError):
            continue
        hits.extend(
            f'{path.relative_to(ROOT)}: {needle}' for needle in FORBIDDEN if needle in text.lower()
        )
    assert not hits, hits
