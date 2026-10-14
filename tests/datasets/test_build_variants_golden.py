"""``build_import_fixture.build_variants`` writes byte-identical layouts and the same
``FIXTURE.json`` for fixed faked annotations. The golden pins the SHA-256 of every
file (labels, yaml, annotation JSON) and the returned fixture dict, so the builder
can be restructured without changing one label row."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.datasets import build_import_fixture as b
from tests.datasets.test_build_import_fixture import _annotations


GOLDEN = Path(__file__).parent / 'golden' / 'build_variants.json'


def _writer(file_name: str, dest: Path, width: int, height: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(f'{file_name} {width}x{height}', encoding='utf-8')


def build(out: Path) -> dict:
    annotations, meta = _annotations()
    fixture = b.build_variants(annotations, meta, out, _writer)
    files = {
        str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.rglob('*'))
        if p.is_file()
    }
    return {'fixture': fixture, 'files': files}


def test_build_variants_output_matches_the_pinned_golden(tmp_path: Path) -> None:
    assert build(tmp_path) == json.loads(GOLDEN.read_text())
